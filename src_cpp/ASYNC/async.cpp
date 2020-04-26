/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 *                                                                           *
 *****************************************************************************

This code is covered by the Community Source License (CPL), version
1.0 as published by IBM and reproduced in the file "license.txt" in the
"license" subdirectory. Redistribution in source and binary form, with
or without modification, is permitted ONLY within the regulations
contained in above mentioned license.

Use of the name and trademark "Intel(R) MPI Benchmarks" is allowed ONLY
within the regulations of the "License for Use of "Intel(R) MPI
Benchmarks" Name and Trademark" as reproduced in the file
"use-of-trademark-license.txt" in the "license" subdirectory.

THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT
LIMITATION, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. Each Recipient is
solely responsible for determining the appropriateness of using and
distributing the Program and assumes all risks associated with its
exercise of rights under this Agreement, including but not limited to
the risks and costs of program errors, compliance with applicable
laws, damage to or loss of data, programs or equipment, and
unavailability or interruption of operations.

EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, NEITHER RECIPIENT NOR
ANY CONTRIBUTORS SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING
WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OR
DISTRIBUTION OF THE PROGRAM OR THE EXERCISE OF ANY RIGHTS GRANTED
HEREUNDER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

EXPORT LAWS: THIS LICENSE ADDS NO RESTRICTIONS TO THE EXPORT LAWS OF
YOUR JURISDICTION. It is licensee's responsibility to comply with any
export regulations applicable in licensee's jurisdiction. Under
CURRENT U.S. export regulations this software is eligible for export
from the U.S. and can be downloaded by or otherwise exported or
reexported worldwide EXCEPT to U.S. embargoed destinations which
include Cuba, Iraq, Libya, North Korea, Iran, Syria, Sudan,
Afghanistan and any other country to which the U.S. has embargoed
goods and services.

 ***************************************************************************
*/

#include "async_benchmark.h"

//!!!
#include <unistd.h>

namespace async_suite {

    inline bool set_stride(int rank, int size, int &stride, int &group)
    {
        if (stride == 0)
            stride = size / 2;
        if (stride <= 0 || stride > size / 2)
            return false;
        group = rank / stride;
        if ((group / 2 == size / (2 * stride)) && (size % (2 * stride) != 0))
            return false;
        return true;
    }


    void AsyncBenchmark::init() {
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(MPI_Datatype, datatype);
        scope = std::make_shared<VarLenScope>(len);
        MPI_Type_size(datatype, &dtsize);
        size_t size_to_alloc = (size_t)scope->get_max_len() * (size_t)dtsize;
        if (size_to_alloc <= ASSUMED_CACHE_SIZE * 3)
            size_to_alloc = ASSUMED_CACHE_SIZE * 3;
        rbuf = (char *)calloc(size_to_alloc, 1);
        sbuf = (char *)calloc(size_to_alloc, 1);
        if (rbuf == nullptr || sbuf == nullptr)
            throw std::runtime_error("AsyncBenchmark: memory allocation error.");
        allocated_size = size_to_alloc;
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    
    void AsyncBenchmark::run(const scope_item &item) { 
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(std::vector<int>, ncycles);
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(int, nwarmup);
        assert(len.size() != 0);
        assert(ncycles.size() != 0);
        int item_ncycles = ncycles[0];
        for (size_t i =0; i < len.size(); i++) {
            if (item.len == (size_t)len[i]) {
                item_ncycles = (i >= ncycles.size() ? ncycles.back() : ncycles[i]);
            }
        }
        double time, tover_comm, tover_calc;
        bool done = benchmark(item.len, datatype, nwarmup, item_ncycles, time, tover_comm, tover_calc);
        if (!done) {
            results[item.len] = result { false, 0.0, 0.0, 0.0, item_ncycles };
        }
    }

    struct YamlOutputMaker {
        const std::string &block;
        YamlOutputMaker(const std::string &_block) : block(_block) {}
        std::map<const std::string, double> kv;
        void add(const std::string &key, double value) { kv[key] = value; }
        void add(int key, double value) { add(std::to_string(key), value); }
        void make_output(YAML::Emitter &yaml_out) const {
            yaml_out << YAML::Key << block << YAML::Value;
            yaml_out << YAML::Flow << YAML::BeginMap;
            for (auto &item : kv) {
                yaml_out << YAML::Key << YAML::Flow << item.first << YAML::Value << item.second;
            }
            yaml_out << YAML::Flow << YAML::EndMap;
        } 
    };

    static void WriteOutYaml(YAML::Emitter &yaml_out, const std::string &bname, 
                            const std::vector<YamlOutputMaker> &makers) {
        yaml_out << YAML::Key << YAML::Flow << bname << YAML::Value;
        yaml_out << YAML::Flow << YAML::BeginMap;
        for (auto &m : makers) {
            m.make_output(yaml_out);
        }
        yaml_out << YAML::Flow << YAML::EndMap;
    }

    void AsyncBenchmark::finalize() { 
        GET_PARAMETER(YAML::Emitter, yaml_out);
        YamlOutputMaker yaml_tmin("tmin");
        YamlOutputMaker yaml_tmax("tmax");
        YamlOutputMaker yaml_tavg("tavg");
        YamlOutputMaker yaml_over_comm("over_comm");
        YamlOutputMaker yaml_over_calc("over_calc");
        YamlOutputMaker yaml_over_full("over_full");
        YamlOutputMaker yaml_topo("topo");
        for (auto it = results.begin(); it != results.end(); ++it) {
            int len = it->first;
            double time = (it->second).time, tmin = 0, tmax = 0, tavg = 0;
            double tover_comm = 0, tover_calc = 0;
            int is_done = ((it->second).done ? 1 : 0), nexec = 0;
            MPI_Reduce(&is_done, &nexec, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 1e32;
            MPI_Reduce(&time, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 0.0;
            MPI_Reduce(&time, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&time, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&((it->second).overhead_comm), &tover_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&((it->second).overhead_calc), &tover_calc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                if (nexec == 0) {
                    std::cout << get_name() << ": " << "{ " << "len: " << len << ", "
                        << " error: \"no successful executions!\"" << " }" << std::endl;
                } else {
                    tavg /= nexec;
                    tover_calc /= nexec;
                    std::cout << get_name() << ": " << "{ " << "len: " << len << ", "
                        << "ncycles: " << (it->second).ncycles << ", "
                        << " time: [ " << tmin << ", " 
                                      << tavg << ", " 
                                      << tmax << " ]" 
                        << ", overhead: [ " << tover_comm << " , " << tover_calc 
                                      << " ] }" << std::endl;
                    yaml_tmin.add(len, tmin);
                    yaml_tmax.add(len, tmax);
                    yaml_tavg.add(len, tavg);
                    yaml_over_comm.add(len, tover_comm); 
                    yaml_over_calc.add(len, tover_calc); 
                    yaml_over_full.add(len, tover_calc + tover_comm);
                }
            }
        }
        yaml_topo.add("np", np);
        WriteOutYaml(yaml_out, get_name(), {yaml_tavg, yaml_over_full, yaml_topo});
    }
    AsyncBenchmark::~AsyncBenchmark() {
        free(rbuf);
        free(sbuf);
    }


    bool AsyncBenchmark_pt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) {
        tover_comm = 0;
	tover_calc = 0;
        int stride = 0, group;
        if (!set_stride(rank, np, stride, group)) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0;
        const int tag = 1;
        int pair = -1;
        if (group % 2 == 0) {
            pair = rank + stride;
            for(int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                MPI_Send((char*)sbuf + (i%n)*b, count, datatype, pair, tag, MPI_COMM_WORLD);
                MPI_Recv((char*)rbuf + (i%n)*b, count, datatype, pair, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } else {
            pair = rank - stride;
            for(int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                MPI_Recv((char*)rbuf + (i%n)*b, count, datatype, pair, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send((char*)sbuf + (i%n)*b, count, datatype, pair, tag, MPI_COMM_WORLD);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0.0, 0.0, ncycles };
        return true;
    }

    void AsyncBenchmark_ipt2pt::init() {
        AsyncBenchmark::init();
        calc.init();
    }

    bool AsyncBenchmark_ipt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) {
        int stride = 0, group;
        if (!set_stride(rank, np, stride, group)) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0, ctime = 0, total_ctime = 0, total_tover_comm = 0, total_tover_calc = 0, 
                                          local_ctime = 0, local_tover_comm = 0, local_tover_calc = 0;
        const int tag = 1;
        int pair = -1;
        MPI_Request request[2];
        calc.reqs = request;
        calc.num_requests = 2;
        if (group % 2 == 0) {
            pair = rank + stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                MPI_Isend((char*)sbuf  + (i%n)*b, count, datatype, pair, tag, MPI_COMM_WORLD, &request[0]);
                MPI_Irecv((char*)rbuf + (i%n)*b, count, datatype, pair, MPI_ANY_TAG, MPI_COMM_WORLD, 
                          &request[1]);
                calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_tover_calc);
                if (i >= nwarmup) {
                    total_ctime += local_ctime;
                    total_tover_comm += local_tover_comm;
                    total_tover_calc += local_tover_calc;
                }
                MPI_Waitall(2, request, MPI_STATUSES_IGNORE);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
            ctime = total_ctime / ncycles;
            tover_comm = total_tover_comm / ncycles;
            tover_calc = total_tover_calc / ncycles;
        } else {
            pair = rank - stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                MPI_Isend((char*)sbuf + (i%n)*b, count, datatype, pair, tag, MPI_COMM_WORLD, &request[0]);
                MPI_Irecv((char*)rbuf + (i%n)*b, count, datatype, pair, MPI_ANY_TAG, MPI_COMM_WORLD, 
                          &request[1]);
                calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_tover_calc);
                if (i >= nwarmup) {
                    total_ctime += local_ctime;
                    total_tover_comm += local_tover_comm;
                    total_tover_calc += local_tover_calc;
                }
                MPI_Waitall(2, request, MPI_STATUSES_IGNORE);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
            ctime = total_ctime / ncycles;
            tover_comm = total_tover_comm / ncycles;
            tover_calc = total_tover_calc / ncycles;
//            total_ctime /= ncycles;
//            total_tover_comm /= ncycles;
//            total_tover_calc /= ncycles;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, time - ctime + tover_comm, tover_calc, ncycles };
//        if (calc.wld == workload_t::CALC_AND_PROGRESS) {
//            std::cout << ">> " << "progress stats: " << calc.total_tests << " " << calc.successful_tests << " " << total_tover << std::endl;
//	}
        return true;
    }

    void barrier(int rank, int np) {
#if 0
        (void)rank;
        (void)np;
        MPI_Barrier(MPI_COMM_WORLD);
#else

        int mask = 0x1;
        int dst, src;
        int tmp = 0;
        for (; mask < np; mask <<= 1) {
            dst = (rank + mask) % np;
            src = (rank - mask + np) % np;
            MPI_Sendrecv(&tmp, 0, MPI_BYTE, dst, 1010,
                         &tmp, 0, MPI_BYTE, src, 1010,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#endif
    }

    bool AsyncBenchmark_allreduce::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) {
        tover_comm = 0;
	tover_calc = 0;
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0;
        for(int i = 0; i < ncycles + nwarmup; i++) {
            if (i >= nwarmup) t1 = MPI_Wtime();
            MPI_Allreduce((char *)sbuf + (i%n)*b, (char *)rbuf + (i%n)*b, count, datatype, MPI_SUM, MPI_COMM_WORLD);
            if (i >= nwarmup) {
                t2 = MPI_Wtime();
                time += (t2 - t1);
            }
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0, 0, ncycles };
        return true;
    }

    void AsyncBenchmark_iallreduce::init() {
        AsyncBenchmark::init();
        calc.init();
    }

    bool AsyncBenchmark_iallreduce::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) {
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0, ctime = 0, total_ctime = 0, total_tover_comm = 0, total_tover_calc = 0,
                                          local_ctime = 0, local_tover_comm = 0, local_tover_calc = 0;
	time = 0;
        MPI_Request request[1];
        calc.reqs = request;
        calc.num_requests = 1;
	MPI_Status  status;
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i >= nwarmup) t1 = MPI_Wtime();
            MPI_Iallreduce((char *)sbuf + (i%n)*b, (char *)rbuf + (i%n)*b, count, datatype, MPI_SUM, MPI_COMM_WORLD, request);
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_tover_calc);
            MPI_Wait(request, &status);
            if (i >= nwarmup) {
                t2 = MPI_Wtime();
                time += (t2 - t1);
                total_ctime += local_ctime;
                total_tover_comm += local_tover_comm;
                total_tover_calc += local_tover_calc;
            }
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
        }
        time /= ncycles;
        //total_ctime /= ncycles;
        //total_tover_comm /= ncycles;
        //total_tover_calc /= ncycles;
        ctime = total_ctime / ncycles;
	tover_comm = total_tover_comm / ncycles;
	tover_calc = total_tover_calc / ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, time - ctime + tover_comm, tover_calc, ncycles };
//        if (calc.wld == workload_t::CALC_AND_PROGRESS || calc.wld == workload_t::CALC) {
//            std::cout << ">> " << "calc time: " << int(total_ctime * 1e6) << " " << int(total_tover * 1e6) << std::endl;
//	}
//        if (calc.wld == workload_t::CALC_AND_PROGRESS) {
//            std::cout << ">> " << "progress stats: " << calc.total_tests << " " << calc.successful_tests << " " << total_tover << std::endl;
//	}
        return true;
    }

    void AsyncBenchmark_calc::init() {
        AsyncBenchmark::init();

        GET_PARAMETER(std::vector<int>, calctime);
        GET_PARAMETER(workload_t, workload);
        wld = workload;
        for (size_t i = 0; i < len.size(); i++) {
            calctime_by_len[len[i]] = (i >= calctime.size() ? (calctime.size() == 0 ? 10000 : calctime[calctime.size() - 1]) : calctime[i]);
        }

        for (int i = 0; i < SIZE; i++) {
            x[i] = y[i] = 0.;
            for (int j=0; j< SIZE; j++) {
                a[i][j] = 1.;
            }
        }
        double timings[3];
        int warmup = 50;
        int Nrep = (50000000 / (2 * SIZE*SIZE)) + 1;
        for (int k = 0; k < 3+warmup; k++) {
            double t1 = MPI_Wtime();
            double tover = 0;
            for (int repeat = 0, cnt=999999; repeat < Nrep; repeat++) {
                if (--cnt == 0) { 
                    double ot1 = MPI_Wtime();
                    if (reqs && num_requests) {
                        for (int r = 0; r < num_requests; r++) {
                            if (!stat[r]) {
                                total_tests++;
                                MPI_Test(&reqs[r], &stat[r], MPI_STATUS_IGNORE);
                                if (stat[r]) {
                                    successful_tests++;
                                }
                            }
                        }
                    }
                    double ot2 = MPI_Wtime();
                    tover += (ot2-ot1);
                } 
                for (int i = 0; i < SIZE; i++) {
                    for (int j = 0; j < SIZE; j++) {
                        for (int k = 0; k < SIZE; k++) {
                            c[i][j] += a[i][k] * b[k][j] + repeat*repeat;
                        }
                    }
                }

            }
            double t2 = MPI_Wtime();
            if (k >= warmup)
                timings[k-warmup] = t2 - t1;
        }
        double tmedian = std::min(timings[0], timings[1]);
        if (tmedian < timings[2])
            tmedian = std::min(std::max(timings[0], timings[1]), timings[2]);
        Nrep = (int)((double)Nrep / (tmedian * 1.0e5) + 0.99);
        int ncalcs_min = 0, ncalcs_max = 0;
        MPI_Allreduce(&Nrep, &ncalcs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&Nrep, &ncalcs_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&Nrep, &ncalcs_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        ncalcs /= np;
//	char node[80];
//	gethostname(node, 80-1);
//	std::cout << ">> node: " << node << " " << Nrep << std::endl;
//        if (rank == 0)
//            std::cout << ">> cper10usec=" << ncalcs << " min/max=" << ncalcs_min << "/" << ncalcs_max << std::endl;
/*
	    double tover = 0;
	    double t1 = MPI_Wtime();
            for (int repeat = 0, cnt=70000; repeat < Nrep * 10; repeat++) {
//		if (--cnt == 0) { if (repeat % 70 == 3) cnt++; }

		if (--cnt == 0) {
		    double ot1 = MPI_Wtime();
		    if (reqs && num_requests) {
			for (int r = 0; r < num_requests; r++) {
			    if (!stat[r]) {
				total_tests++;
				MPI_Test(&reqs[r], &stat[r], MPI_STATUS_IGNORE);
				if (stat[r]) {
				    successful_tests++;
				}
			    }
			}
		    }
	            double ot2 = MPI_Wtime();
		    tover += (ot2-ot1);

		} 
                for (int i = 0; i < SIZE; i++) {
                    for (int j = 0; j < SIZE; j++) {
                        for (int k = 0; k < SIZE; k++) {
                            c[i][j] += a[i][k] * b[k][j] + repeat*repeat;
			}
                    }
                }
	    }
            double t2 = MPI_Wtime();
	    if (rank == 0)
		std::cout << ">> TIME=" << int((t2-t1)*1e6) << std::endl;
*/
    }

    bool AsyncBenchmark_calc::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) {
	GET_PARAMETER(int, cper10usec);
	int real_cper10usec;
        (void)datatype;
        total_tests = 0;
        successful_tests = 0;
	time = 0;
        tover_comm = 0;
	tover_calc = 0;
        if (wld == workload_t::NONE) {
            time = 0;
            return true;
        }
        //do_probe = true;
        double t1 = 0, t2 = 0;
        double ot1 = 0, ot2 = 0;
        int R = calctime_by_len[count] * ncalcs / 10;
        if (wld == workload_t::CALC_AND_PROGRESS && reqs) {
            for (int r = 0; r < num_requests; r++) {
                stat[r] = 0;
            }
        }
	if (wld == workload_t::CALC_AND_PROGRESS) {
		for (int i = 0; i < ncycles + nwarmup; i++) {
		    if (i == nwarmup) t1 = MPI_Wtime();
		    for (int repeat = 0, cnt = 90; repeat < R; repeat++) {
#if 1                
			if (--cnt == 0) {
#if 1 
			    ot1 = MPI_Wtime();
			    if (reqs && num_requests) {
				for (int r = 0; r < num_requests; r++) {
				    if (!stat[r]) {
					total_tests++;
					MPI_Test(&reqs[r], &stat[r], MPI_STATUS_IGNORE);
					if (stat[r]) {
					    successful_tests++;
					}
				    }
				}
			    }
			    /*
			    int avail;
			    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &avail, MPI_STATUS_IGNORE);
			    if (avail) {
				std::cout << ">> OK: Iprobe" << std::endl;
				do_probe = false;
			    }
			    */
			    ot2 = MPI_Wtime();
			    tover_comm += (ot2 - ot1);
#endif
			    cnt = 90;
			}
#endif                
			for (int i = 0; i < SIZE; i++) {
			    for (int j = 0; j < SIZE; j++) {
				for (int k = 0; k < SIZE; k++) {
				    c[i][j] += a[i][k] * b[k][j] + repeat * repeat;
				}
			    }
			}

		    }
		}
	} else {
	    for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                for (int repeat = 0, cnt = 700000; repeat < R; repeat++) {

		   // if (--cnt == 0) { if (repeat%70 == 3) cnt++; }
 		    if (--cnt == 0) {
		    double ot1 = MPI_Wtime();
		    if (reqs && num_requests) {
			for (int r = 0; r < num_requests; r++) {
			    if (!stat[r]) {
				total_tests++;
				MPI_Test(&reqs[r], &stat[r], MPI_STATUS_IGNORE);
				if (stat[r]) {
				    successful_tests++;
				}
			    }
			}
		    }
	            double ot2 = MPI_Wtime();
		    tover_comm += (ot2-ot1);

		} 
                    for (int i = 0; i < SIZE; i++) {
                        for (int j = 0; j < SIZE; j++) {
                            for (int k = 0; k < SIZE; k++) {
                                c[i][j] += a[i][k] * b[k][j] + repeat * repeat;
		    	    }
                        }
                    }
                }
	    }
	}
        t2 = MPI_Wtime();
        time = (t2 - t1);
	
	int pure_calc_time = int((time - tover_comm) * 1e6);
	if (!pure_calc_time)
	    return true;
	real_cper10usec = R * 10 / pure_calc_time;
//	if (rank == 0) {
	    if (cper10usec) {
		int R0 = pure_calc_time * cper10usec / 10;
		tover_calc = (double)(R0 - R) / (double)real_cper10usec * 1e-5;
//tover_calc = (time - tover_comm) - (((double)R / (double)cper10usec) * 1e-5);
//  	        std::cout << ">> real_cper10usec=" << real_cper10usec << " cper10usec=" << cper10usec << " tover_calc=" << int(tover_calc * 1e6) << "/" << pure_calc_time << std::endl;
	    } else {
		tover_calc = 0;
//                std::cout << ">> real_cper10usec=" << real_cper10usec << std::endl;
	    }
//	} 
        return true;
    }

    DECLARE_INHERITED(AsyncBenchmark_pt2pt, sync_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_ipt2pt, async_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_allreduce, sync_allreduce)
    DECLARE_INHERITED(AsyncBenchmark_iallreduce, async_allreduce)
    DECLARE_INHERITED(AsyncBenchmark_calc, async_calc)
}
