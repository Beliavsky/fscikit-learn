program test_neighbor_gram
 use sklearn_common,only:dp,require
 use sklearn_neighbor_gram,only:neighbor_row_norm,neighbor_middle_term
 use sklearn_neighbor_functions,only:kneighbors_graph
 use sklearn_sparse,only:csr_matrix
 implicit none
 real(dp)::u(16),x(1,16),norm,middle
 type(csr_matrix)::graph
 ! Actual scipy OpenBLAS 0.3.28 through sklearn's _dot/_gemm wrappers.
 u=[-.8795760690874841_dp,-.2432477723209184_dp,.6740307397120452_dp,.3449463205997444_dp, &
  -.6252848420573858_dp,-.20645163259513075_dp,-.8715974190766754_dp,-.870105814372861_dp, &
  .39814763825715854_dp,-.6652552234417677_dp,.1336772436581866_dp,.786214675313658_dp, &
  -.0053172121614025025_dp,-.5465660754572188_dp,.5545796309718483_dp,.46951714473235734_dp]
 norm=neighbor_row_norm(u);middle=neighbor_middle_term(u,u,1,1,1,1)
 call require(norm==5.420355380866347_dp,'Source neighbor BLAS norm')
 call require(middle==-10.840710761732693_dp,'Source neighbor BLAS middle term')
 call require(norm/=-middle/2,'distinct norm and GEMM accumulation retained')
 x(1,:)=u;graph=kneighbors_graph(x,1,mode='distance',include_self=.true.)
 call require(abs(graph%data(1)-4.2146848510894035e-8_dp)<1.e-22_dp,'Source nonzero diagonal rounding')
 print *,'Neighbor norm, GEMM accumulation and diagonal rounding passed'
end program
