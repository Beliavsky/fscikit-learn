program test_mds
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(mds) :: model,repeat
  real(dp) :: d(3,3),init(3,1),x(15,2),t,stress
  real(dp), allocatable :: z(:,:),again(:,:),dissimilarities(:,:)
  integer :: i
  d=reshape([0._dp,1._dp,3._dp,1._dp,0._dp,2._dp,3._dp,2._dp,0._dp],[3,3])
  init(:,1)=[-1._dp,0._dp,1._dp]; model%dissimilarity='precomputed'; model%max_iter=1
  z=model%fit_transform(d,init)
  call require(size(z,2)==1,'MDS explicit initialization determines dimension')
  call require(maxval(abs(z(:,1)-[-4._dp/3,-1._dp/3,5._dp/3]))<1.e-12_dp,'MDS Guttman update')
  call require(abs(model%stress_-2)<1.e-12_dp,'MDS sklearn 1.6.1 pre-update stress convention')
  model%max_iter=300; model%eps=1.e-10_dp; z=model%fit_transform(d,init)
  call require(model%stress_<1.e-20_dp,'metric MDS recovers Euclidean dissimilarities')
  call require(abs(sum(z))<1.e-12_dp,'metric MDS centering')
  do i=1,15
    t=real(i,dp)/15; x(i,:)=[sin(4*t),cos(7*t)]
  end do
  model%dissimilarity='euclidean'; model%random_state=42; model%n_init=1
  z=model%fit_transform(x); stress=model%stress_
  repeat=model; again=repeat%fit_transform(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'MDS seeded repeatability')
  model%n_init=4; z=model%fit_transform(x)
  call require(model%stress_<=stress+1.e-12_dp,'MDS selects best random initialization')
  dissimilarities=pairwise_distances(x,x); model%dissimilarity='precomputed'
  model%metric=.false.; model%eps=1.e-7_dp
  z=model%fit_transform(dissimilarities,x)
  again=model%fit_transform(dissimilarities**2,x)
  call require(maxval(abs(z-again))<1.e-10_dp,'nonmetric MDS depends on dissimilarity order')
  call require(model%stress_>=0.and.model%stress_<1,'nonmetric normalized stress range')
  model%metric=.true.; d=0; z=model%fit_transform(d)
  call require(maxval(abs(z))==0.and.model%stress_==0,'zero metric dissimilarities handled without division')
  print *, 'multidimensional scaling: PASS'
end program
