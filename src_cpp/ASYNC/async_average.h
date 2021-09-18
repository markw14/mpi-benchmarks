#pragma once

static inline double get_avg(double x, int nexec, int rank, int np, bool is_done) {
	double xx = x;
	std::vector<double> fromall;
	if (rank == 0)
		fromall.resize(np);
	if (!is_done) 
		xx = 0;
	MPI_Gather(&xx, 1, MPI_DOUBLE, fromall.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank != 0)
		return 0;
	const char *avg_option = nullptr;
	if (!(avg_option = getenv("IMB_ASYNC_AVG_OPT"))) {
		avg_option = "MEDIAN";
	}
	if (std::string(avg_option) == "MEDIAN") {
		std::sort(fromall.begin(), fromall.end());
		if (nexec == 0)
			return 0;
		int off = np - nexec;
		if (nexec == 1)
			return fromall[off];
		if (nexec == 2) {
			return (fromall[off] + fromall[off+1]) / 2.0;
		}
		return fromall[off + nexec / 2];
	}
	if (std::string(avg_option) == "AVERAGE") {
		double sum = 0;
		for (auto x : fromall)
			sum += x;
		sum /= fromall.size();
		return sum;
	}
	if (std::string(avg_option) == "MAX") {
		double maxx = 0;
		for (auto x : fromall)
			maxx = std::max(x, maxx);
		return maxx;
	}
	return -1;
}

