program test_common
  use sklearn_common
  implicit none
  real(dp) :: a(2,2), b(2), v(2), q(2,2)
  real(dp) :: wide(3,5),s(5),u(3,5),right(5,5),zero(3,5)
  a=reshape([2._dp,1._dp,1._dp,2._dp],[2,2]); b=[3._dp,3._dp]
  call require(maxval(abs(solve_system(a,b)-1))<1.e-12_dp,'linear solve')
  call symmetric_eigen(a,v,q)
  call require(maxval(abs(v-[3._dp,1._dp]))<1.e-12_dp,'eigenvalues')
  call require(maxval(abs(matmul(a,q)-q*spread(v,1,2)))<1.e-12_dp,'eigenvectors')
  call require(all(unique_labels([3,-1,3,7])==[-1,3,7]),'sorted unique labels')
  call require(abs(sigmoid(-1000._dp))<tiny(1._dp),'stable sigmoid')
  call seed_rng(7); wide=normal_random(3,5); call dense_svd(wide,s,u,right)
  call require(maxval(abs(wide-matmul(u*spread(s,1,3),transpose(right))))<1.e-12_dp,'wide SVD')
  zero=0; call dense_svd(zero,s,u,right)
  call require(all(s==0).and.all(u==0),'zero SVD')
  print *, 'common: PASS'
end program
