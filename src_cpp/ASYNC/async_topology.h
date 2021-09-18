#pragma once

#include <memory>

enum action_t {
    SEND = 0, RECV = 1
};

struct peer_t {
    int rank;
    action_t action;
};

using actions_t = std::vector<peer_t>;

struct topohelper {
    int np, rank;
    bool active, bidirectional;

    topohelper(int np_, int rank_) : np(np_), rank(rank_) {}

    static std::shared_ptr<topohelper> create(const params::list<params::benchmarks_params> &pl, 
                                              int np_, int rank_);
	virtual bool is_active() = 0;

	virtual bool is_even() { return true; }

    virtual int get_group() { return 0; }

    std::vector<int> ranks_for_action(action_t action) {
        auto c = comm_actions();
        std::vector<int> result;
        std::set<int> r;
        for (auto &p : c) {
            if (p.action == action)
                r.insert(p.rank);
        }
        std::copy(r.begin(), r.end(), std::back_inserter(result));
        return result;

    }

    virtual std::vector<int> ranks_to_send_to() {
        return ranks_for_action(action_t::SEND);
	}

    virtual std::vector<int> ranks_to_recv_from() {
        return ranks_for_action(action_t::RECV);
	}

    virtual actions_t comm_actions() = 0;
    virtual size_t get_num_actions() = 0;
};

struct topo_pingpong : public topohelper {
	int stride, group;
    bool set_stride() {
        if (stride == 0)
            stride = np / 2;
        if (stride <= 0 || stride > np / 2)
            return false;
        group = rank / stride;
        if ((group / 2 == np / (2 * stride)) && (np % (2 * stride) != 0))
            return false;
        return true;
    }

	topo_pingpong(const params::list<params::benchmarks_params> &pl, int np_, int rank_) : 
                                                                            topohelper(np_, rank_) {
		stride = pl.get_int("stride");
		bidirectional = pl.get_bool("bidirectional");
		active = set_stride();
    }

	virtual bool is_active() override { return active; }

	virtual bool is_even() override { return group % 2; }

    virtual int get_group() { return group; }

    virtual actions_t comm_actions() override {
        actions_t result;
        if (is_even()) {
            result.push_back({prev(), action_t::SEND});
            if (bidirectional)
                result.push_back({prev(), action_t::RECV});
        } else {
            result.push_back({next(), action_t::RECV});
            if (bidirectional)
                result.push_back({next(), action_t::SEND});
        }
        return result;
    }

    virtual size_t get_num_actions() override {
        return ranks_to_send_to().size() + ranks_to_recv_from().size();
    }

    protected:
	int prev() {
			int distance = stride % np;
			int p = rank - distance;
			if (p < 0)
					p += np;
			return p;
	}

	int next() {
			int distance = stride % np;
			int n = rank + distance;
			if (n >= np)
					n -= np;
			return n;
	}
};

struct topo_split : public topohelper {
    int nparts, active_parts;
    int group;

    bool handle_split() {
        group = rank % nparts;
        int rest = np % nparts;
        if (rank >= np - rest) {
            return false;
        }
        return true;
    }

    topo_split(const params::list<params::benchmarks_params> &pl, int np_, int rank_) :
                                                                  topohelper(np_, rank_) {
        nparts = pl.get_int("nparts");
        active_parts = pl.get_int("active_parts");
        if (!active_parts) {
            active_parts = nparts;
        }
        active = handle_split();
    }

	virtual bool is_active() override { 
        return active; 
    }

    virtual int get_group() { 
        return group; 
    }
    
    virtual std::vector<int> ranks_to_send_to() override {
        std::vector<int> result; 
        for (int n = group; n < np / nparts; n++) {
            if (n != rank)
                result.push_back(n);
        }
        return result;
    }

    virtual std::vector<int> ranks_to_recv_from() override { 
        return ranks_to_send_to(); 
    }

    virtual actions_t comm_actions() override {
        actions_t result;
        for (auto r : ranks_to_recv_from()) {
            result.push_back({r, action_t::RECV});
            result.push_back({r, action_t::SEND});
        }
        return result;
    }

    size_t get_num_actions() override { 
        return comm_actions().size(); 
    }
};

struct topo_neighb : public topohelper {
    int nneighb;
	topo_neighb(const params::list<params::benchmarks_params> &pl, int np_, int rank_) : topohelper(np_, rank_) {
		nneighb = pl.get_int("nneighb");
		bidirectional = pl.get_bool("bidirectional");
        active = (np > nneighb);
    }

    virtual bool is_even() { return rank % 2; }

	virtual bool is_active() override { return active; }

    virtual actions_t comm_actions() override {
        actions_t result;
        if (is_even()) {
            for (int i = 0; i < nneighb; i++) {
                result.push_back({next(i), action_t::SEND});
                result.push_back({prev(i), action_t::RECV});
                if (bidirectional) {
                    result.push_back({next(i), action_t::RECV});
                    result.push_back({prev(i), action_t::SEND});
                }
            }
        } else {
            for (int i = 0; i < nneighb; i++) {
                result.push_back({prev(i), action_t::RECV});
                result.push_back({next(i), action_t::SEND});
                if (bidirectional) {
                    result.push_back({prev(i), action_t::SEND});
                    result.push_back({next(i), action_t::RECV});
                }
            }
        }
        return result;
    }

    virtual size_t get_num_actions() override {
        return comm_actions().size();
    }

    int prev(int i) {
		int distance = (i + 1) % np;
		int p = rank - distance;
		if (p < 0)
			p += np;
		return p;
	}

	int next(int i) {
		int distance = (i + 1) % np;
		int n = rank + distance;
		if (n >= np)
			n -= np;
		return n;
	}
};

std::shared_ptr<topohelper> topohelper::create(const params::list<params::benchmarks_params> &pl,
                                               int np_, int rank_) {
    if (pl.get_string("topology") == "ping-pong") {
        return std::make_shared<topo_pingpong>(pl, np_, rank_);
    } else if (pl.get_string("topology") == "split") {
        return std::make_shared<topo_split>(pl, np_, rank_);
    } else if (pl.get_string("topology") == "neighb") {
        return std::make_shared<topo_neighb>(pl, np_, rank_);
	}
    throw std::runtime_error("topohelper: not supported topology in creator");
}
