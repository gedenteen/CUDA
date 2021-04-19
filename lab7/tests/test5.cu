#include <thrust/tuple.h>
#include <stdio.h>

int main(){
	thrust::tuple<int, int, float, const char *> test_tuple(23, 99, 4.5, "thrust");
	printf("%d\t%d\t%g\t%s\n", thrust::get<0>(test_tuple),
		thrust::get<1>(test_tuple),
		thrust::get<2>(test_tuple),
		thrust::get<3>(test_tuple));
	return 0;
}
