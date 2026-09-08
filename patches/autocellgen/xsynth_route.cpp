////////////////////////////////////////////////////////////////////////////
// X-Synth routing-only entry point  (patch P1)
//
// This file is owned by the X-Synth repository; scripts/build_backend.sh copies
// it into the AutoCellGen submodule's src/. Keeping it here rather than editing
// the submodule in place means it lives in our own history.
//
// Why this exists
// -----------
// Upstream main.cpp routes the placer's candidates in the placer's own order.
// X-Synth has to route in the order the predictor and ranker choose, so "route
// this one placement candidate" must be callable from outside: the R interface
// of paper §2.1.
//
// Usage
//   placement --xsynth-route <placement.txt> --cell <name> \
//             -i <netlist.sp> -d <style> [--order 3,1,7] [--out <dir>]
//
// Without --order every candidate is routed in file order.
// Results go to stdout as one JSON line each, so summary.txt need not be parsed.
//
// WARNING on index conventions: placement files count from 1
//    ("-------- Solution 1 --------") while `Solution k` in summary.txt counts
//    from 0. --order is 0-based, to match summary.txt.
////////////////////////////////////////////////////////////////////////////

#include "../header/global.h"
#include "../header/cdlParser.h"
#include "../header/PlaceGrid.h"
#include "../header/Router.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Defined in main.cpp; not static, so it links.
void parsing_DR(fs::path &dr_path);

namespace {

std::string arg_of(int argc, char **argv, const char *flag) {
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return "";
}

// Pull one side out of a line like
// "NMOS : MM0(3) [VSS A Y], PMOS : MM1(3) [VDD A Y]".
bool parse_side(const std::string &line, const std::string &tag, OutputTrans &out) {
    size_t p = line.find(tag + " : ");
    if (p == std::string::npos) return false;
    p += tag.size() + 3;

    size_t lp = line.find('(', p);
    size_t rp = line.find(')', lp);
    size_t lb = line.find('[', rp);
    size_t rb = line.find(']', lb);
    if (lp == std::string::npos || rp == std::string::npos ||
        lb == std::string::npos || rb == std::string::npos) return false;

    std::string name = line.substr(p, lp - p);
    int nfin = std::atoi(line.substr(lp + 1, rp - lp - 1).c_str());

    std::istringstream terms(line.substr(lb + 1, rb - lb - 1));
    std::string left, gate, right;
    if (!(terms >> left >> gate >> right)) return false;

    out.set(name, left, gate, right, nfin);
    return true;
}

// Turn a placement file written by the placer back into a list of PlaceGrids.
std::vector<PlaceGrid> parse_placements(const fs::path &path) {
    std::vector<PlaceGrid> out;
    std::ifstream in(path.string());
    if (!in) return out;

    std::vector<OutputTrans> nmos, pmos;
    auto flush = [&]() {
        if (!nmos.empty()) {
            out.emplace_back(nmos, pmos, static_cast<int>(nmos.size()));
            nmos.clear();
            pmos.clear();
        }
    };

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("-------- Solution", 0) == 0) { flush(); continue; }
        if (line.rfind("NMOS :", 0) != 0) continue;

        OutputTrans n("", "", "", "", 0), p("", "", "", "", 0);
        if (parse_side(line, "NMOS", n) && parse_side(line, "PMOS", p)) {
            nmos.push_back(n);
            pmos.push_back(p);
        }
    }
    flush();
    return out;
}

std::vector<int> parse_order(const std::string &spec, int n) {
    std::vector<int> order;
    if (spec.empty()) {
        for (int i = 0; i < n; i++) order.push_back(i);
        return order;
    }
    std::istringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        int k = std::atoi(tok.c_str());
        if (k >= 0 && k < n) order.push_back(k);
        else std::cerr << "xsynth: ignoring out-of-range index: " << k << std::endl;
    }
    return order;
}

// P7: the solver verdict. unsat (unroutable) and unknown (the solver gave up)
// are different things.
const char *solver_name(int st) {
    switch (st) {
        case 0: return "unsat";
        case 1: return "sat";
        case 2: return "unknown";
        default: return "not_run";
    }
}

std::string json_escape(const std::string &s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') r += '\\';
        r += c;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Tile hints (paper §2.2)
//
// A PlaceGrid is fully determined by nmos/pmos/cellWidth, so a column slice is
// exactly a tile. The Router only reads nets and IOnets from a Cell
// (Router.cpp:87,92,374), so handing it a partial Cell holding just the nets in
// the window routes that window locally.
//
//   IOnets kept  -> y_route (pin-access-aware)
//   IOnets empty -> y_naive (pin-unaware)
//
// This maps directly onto the paper's hint vector h_i = (y_route, y_naive, m1, m2, v).
// ---------------------------------------------------------------------------

// Collect the nets appearing in the window [col0, col0+w).
std::vector<std::string> window_nets(const PlaceGrid &pg, int col0, int w) {
    std::vector<std::string> nets;
    auto add = [&](const std::string &n) {
        if (n.empty() || n == "dummy") return;
        if (std::find(nets.begin(), nets.end(), n) == nets.end()) nets.push_back(n);
    };
    for (int i = col0; i < col0 + w; i++) {
        add(pg.nmos[i].left);  add(pg.nmos[i].gate);  add(pg.nmos[i].right);
        add(pg.pmos[i].left);  add(pg.pmos[i].gate);  add(pg.pmos[i].right);
    }
    return nets;
}

PlaceGrid slice(const PlaceGrid &pg, int col0, int w) {
    std::vector<OutputTrans> n(pg.nmos.begin() + col0, pg.nmos.begin() + col0 + w);
    std::vector<OutputTrans> p(pg.pmos.begin() + col0, pg.pmos.begin() + col0 + w);
    return PlaceGrid(n, p, w);
}

int xsynth_tiles_main(int argc, char **argv) {
    std::string place_file = arg_of(argc, argv, "--xsynth-tiles");
    if (place_file.empty()) return -1;

    std::string cell_name = arg_of(argc, argv, "--cell");
    std::string netlist = arg_of(argc, argv, "-i");
    std::string style = arg_of(argc, argv, "-d");
    std::string order_spec = arg_of(argc, argv, "--order");
    std::string out_dir = arg_of(argc, argv, "--out");
    std::string sw = arg_of(argc, argv, "--tile-w");
    std::string ss = arg_of(argc, argv, "--tile-s");
    if (out_dir.empty()) out_dir = "xsynth_tiles_out";
    int tw = sw.empty() ? 7 : std::atoi(sw.c_str());   // the paper's default w=7
    int ts = ss.empty() ? 2 : std::atoi(ss.c_str());   //           s=2

    if (cell_name.empty() || netlist.empty() || style.empty() || tw <= 0 || ts <= 0) {
        std::cerr << "usage: placement --xsynth-tiles <placement.txt> --cell <name> "
                     "-i <netlist.sp> -d <style> [--tile-w 7] [--tile-s 2] "
                     "[--order 0,1] [--out <dir>]" << std::endl;
        return 2;
    }

    // A narrow window may contain no access point for an external pin. Upstream
    // calls exit(0) there, so patch P6's graceful failure is enabled. That is
    // the paper's v (valid) flag.
    setenv("XSYNTH_NO_EXIT", "1", 1);

    fs::path dr_path(style);
    parsing_DR(dr_path);

    Library lib;
    cdlParser parser(lib, fs::path(netlist));
    parser.parse();

    Cell *cell = nullptr;
    for (auto &c : lib.cells)
        if (c.name == cell_name) { cell = &c; break; }
    if (!cell) {
        std::cerr << "xsynth: cell not in the netlist: " << cell_name << std::endl;
        return 3;
    }

    std::vector<PlaceGrid> cands = parse_placements(fs::path(place_file));
    if (cands.empty()) {
        std::cerr << "xsynth: could not parse any placement candidate" << std::endl;
        return 4;
    }

    fs::create_directories(out_dir);
    std::vector<int> order = parse_order(order_spec, static_cast<int>(cands.size()));

    for (int k : order) {
        const PlaceGrid &full = cands[k];

        // Paper §2.2: a window never crosses the cell boundary. A cell narrower
        // than the window has no tiles.
        if (full.cellWidth < tw) {
            std::cout << "{\"cell\":\"" << json_escape(cell_name) << "\",\"k\":" << k
                      << ",\"tile\":-1,\"valid\":false"
                      << ",\"reason\":\"cellWidth<tile_w\"}" << std::endl;
            continue;
        }

        int tile = 0;
        for (int col0 = 0; col0 + tw <= full.cellWidth; col0 += ts, tile++) {
            PlaceGrid pg = slice(full, col0, tw);
            pg.cal_cost();

            std::vector<std::string> nets = window_nets(full, col0, tw);

            // Nets in the window that are external pins of the cell
            std::vector<std::string> io;
            for (const auto &n : nets)
                if (std::find(cell->IOnets.begin(), cell->IOnets.end(), n) !=
                    cell->IOnets.end()) io.push_back(n);

            bool io_fail = false;
            auto route_once = [&](bool pin_aware, double &m1, double &m2,
                                  long long &rt) -> bool {
                Cell sub;
                sub.name = cell_name;
                sub.nets = nets;
                if (pin_aware) sub.IOnets = io;   // empty means pin-unaware
                fs::path rp = fs::path(out_dir) /
                              fs::path(cell_name + "_k" + std::to_string(k) +
                                       "_t" + std::to_string(tile) +
                                       (pin_aware ? "_pa" : "_nv") + ".txt");
                Router r(sub, pg);
                r.routing(rp);
                m1 = r.m1_usage;
                m2 = r.m2_usage;
                rt = r.runtime;
                if (r.io_error) io_fail = true;   // the v flag of the paper's hint vector
                return r.is_routable;
            };

            double m1 = 0, m2 = 0, m1n = 0, m2n = 0;
            long long rt = 0, rtn = 0;
            bool y_route = route_once(true, m1, m2, rt);
            bool y_naive = route_once(false, m1n, m2n, rtn);

            std::cout << "{\"cell\":\"" << json_escape(cell_name) << "\""
                      << ",\"k\":" << k
                      << ",\"tile\":" << tile
                      << ",\"col0\":" << col0
                      << ",\"tile_w\":" << tw
                      << ",\"y_route\":" << (y_route ? "true" : "false")
                      << ",\"y_naive\":" << (y_naive ? "true" : "false")
                      << ",\"m1\":" << m1
                      << ",\"m2\":" << m2
                      << ",\"n_io\":" << io.size()
                      << ",\"runtime_ms\":" << (rt + rtn)
                      << ",\"valid\":" << (io_fail ? "false" : "true")
                      << "}" << std::endl;
        }
    }
    return 0;
}

}  // namespace

// Called at the top of main(). Returns -1 when --xsynth-route is absent, so the
// upstream flow continues unchanged.
int xsynth_route_main(int argc, char **argv) {
    // Tile-hint mode takes precedence. With neither flag, return -1 for upstream.
    int t = xsynth_tiles_main(argc, argv);
    if (t >= 0) return t;

    std::string place_file = arg_of(argc, argv, "--xsynth-route");
    if (place_file.empty()) return -1;

    std::string cell_name = arg_of(argc, argv, "--cell");
    std::string netlist = arg_of(argc, argv, "-i");
    std::string style = arg_of(argc, argv, "-d");
    std::string order_spec = arg_of(argc, argv, "--order");
    std::string out_dir = arg_of(argc, argv, "--out");
    if (out_dir.empty()) out_dir = "xsynth_route_out";

    if (cell_name.empty() || netlist.empty() || style.empty()) {
        std::cerr << "usage: placement --xsynth-route <placement.txt> --cell <name> "
                     "-i <netlist.sp> -d <style> [--order 3,1,7] [--out <dir>]"
                  << std::endl;
        return 2;
    }

    fs::path dr_path(style);
    parsing_DR(dr_path);   // fills the global `setting`, which Router depends on

    Library lib;
    cdlParser parser(lib, fs::path(netlist));
    parser.parse();

    Cell *cell = nullptr;
    for (auto &c : lib.cells)
        if (c.name == cell_name) { cell = &c; break; }
    if (!cell) {
        std::cerr << "xsynth: cell not in the netlist: " << cell_name << std::endl;
        return 3;
    }

    std::vector<PlaceGrid> cands = parse_placements(fs::path(place_file));
    if (cands.empty()) {
        std::cerr << "xsynth: could not parse any placement candidate: " << place_file
                  << std::endl;
        return 4;
    }

    fs::create_directories(out_dir);
    std::vector<int> order = parse_order(order_spec, static_cast<int>(cands.size()));

    for (int k : order) {
        PlaceGrid pg = cands[k];
        pg.cal_cost();   // h_net_den / cong / max_* / cost, derived values Router reads

        fs::path rp = fs::path(out_dir) /
                      fs::path(cell_name + "_k" + std::to_string(k) + ".txt");

        Router router(*cell, pg);
        router.routing(rp);

        // One JSON line, so summary.txt need not be parsed.
        std::cout << "{\"cell\":\"" << json_escape(cell_name) << "\""
                  << ",\"k\":" << k
                  << ",\"width\":" << pg.cellWidth
                  << ",\"cost\":" << pg.cost
                  << ",\"routable\":" << (router.is_routable ? "true" : "false")
                  << ",\"m1\":" << router.m1_usage
                  << ",\"m2\":" << router.m2_usage
                  << ",\"runtime_ms\":" << router.runtime
                  << ",\"solver\":\"" << solver_name(router.solver_status) << "\""
                  << "}" << std::endl;
    }
    return 0;
}
