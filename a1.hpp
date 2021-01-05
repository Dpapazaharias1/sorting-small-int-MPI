/*  Demetrios
 *  Papazaharias
 *  dvpapaza
 */

#ifndef A1_HPP
#define A1_HPP

#include <vector>

void isort(std::vector<short int> &x, MPI_Comm comm)
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Compute the number of times element occurs in each subarray X_i. O(n/p)

	short int max_value = 32000;
	short int min_value = -32000;
	int num_int = (max_value - min_value) + 1; // 64,001 values for short int
	std::vector<long long int> count(num_int); // Must be long long int for edge case where list contains n/p of same value

	for (const auto &e : x)
	{
		++count[e - min_value];
	};

	// Make sure nodes are synchronized before we perform Allreduce;
	MPI_Barrier(comm);

	// ---- MPI_Reduce to obtain global histogram

	std::vector<long long int> buf(num_int, 0);

	MPI_Allreduce(count.data(), buf.data(), num_int, MPI_LONG_LONG_INT, MPI_SUM, comm);
	MPI_Barrier(comm);

	long long int n = x.size(); // Store number of keys that belong to rank p
	long long int n0;			// Extra broadcast step, we need to know the size at 0 just incase we don't have an even split.

	if (rank == 0)
	{
		n0 = x.size();
	}

	MPI_Barrier(comm);
	MPI_Bcast(&n0, 1, MPI_LONG_LONG_INT, 0, comm);

	long long int start_idx = 0;
	if (rank != 0)
	{
		start_idx = n0 + (n * (rank - 1));
	}
	long long int end_idx = start_idx + n;

	std::vector<long long int> prefix_result(num_int, 0);
	std::partial_sum(buf.begin(), buf.end(), prefix_result.begin());

	short int start_k = min_value;
	short int end_k = max_value;

	// Find the starting and ending keys for each proccessor. Used a linear search O(k)

	for (short int k = min_value; k < max_value; ++k)
	{
		if (
			start_idx <= prefix_result[k - min_value] &&
			start_idx >= prefix_result[k - min_value] - buf[k - min_value])
		{
			start_k = k;
			break;
		}
	}
	for (short int k = start_k; k < max_value; ++k)
	{
		if (
			end_idx <= prefix_result[k - min_value] &&
			end_idx >= prefix_result[k - min_value] - buf[k - min_value])
		{
			end_k = k;
			break;
		}
	}

	// Fill x with sorted values, O(n/p)
	long long int idx = 0;
	// k = start_k
	for (int i = 0; i < prefix_result[start_k - min_value] - start_idx; ++i)
	{
		x[idx++] = start_k;
	}
	// k = start_k + 1, ... , end_k - 1
	for (short int k = start_k + 1; k < end_k; ++k)
	{
		for (int i = 0; i < buf[k - min_value]; i++)
		{
			x[idx++] = k;
		}
	}
	// k = end_k
	while (idx < (end_idx - start_idx))
	{
		x[idx++] = end_k;
	}

} // isort

#endif // A1_HPP
