program test_numpy_ordering
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort,numpy_argpartition
 implicit none
 real(dp)::x(9)
 integer::ids(9)
 x=[1._dp,0._dp,1._dp,0._dp,0._dp,0._dp,0._dp,0._dp,0._dp]
 ids=numpy_argsort(x);call require(all(x(ids(2:))>=x(ids(:8))),'sorted numeric keys')
 ids=numpy_argpartition(x,9);call require(ids(9)==1,'NumPy AVX512 farthest tie choice')
 ids=numpy_argpartition(x,4)
 call require(maxval(x(ids(:3)))<=x(ids(4)).and.minval(x(ids(5:)))>=x(ids(4)),'partition inequality')
 print *, 'NumPy index ordering tests PASS'
end program
