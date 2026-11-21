program test_nca
  use sklearn_nca
  use sklearn_common, only: dp,require
  implicit none
  type(neighborhood_components_analysis) :: model,copy
  real(dp) :: x(12,3),start(2,3),g(2,3),plus(2,3),minus(2,3),f,fp,fm,work(2,3)
  real(dp), allocatable :: z(:,:),original(:,:),before(:,:)
  integer :: y(12),i,j
  do i=1,12
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),real(mod(i,3),dp)]
    y(i)=mod(i,3)-1
  end do
  original=x;start=reshape([.7_dp,.1_dp,.2_dp,.6_dp,-.1_dp,.3_dp],[2,3])
  call nca_objective_gradient(x,y,start,f,g)
  do i=1,2
    do j=1,3
      plus=start;minus=start;plus(i,j)=plus(i,j)+1.e-6_dp;minus(i,j)=minus(i,j)-1.e-6_dp
      call nca_objective_gradient(x,y,plus,fp,work)
      call nca_objective_gradient(x,y,minus,fm,work)
      call require(abs((fp-fm)/2.e-6_dp-g(i,j))<1.e-7_dp,'NCA analytic gradient finite differences')
    end do
  end do
  model%init_matrix=start;model%n_components=2;model%max_iter=80
  z=model%fit_transform(x,y)
  call require(model%objective_>=f,'NCA training improves leave-one-out objective')
  call require(all(shape(z)==[12,2]),'NCA transform shape')
  call require(all(x==original),'NCA preserves caller input')
  call require(model%n_iter_==model%n_optimizer_iter_+1,'NCA source callback iteration count')
  copy=model;before=model%components_;copy%warm_start=.true.;copy%max_iter=3
  call copy%fit(x,y)
  call require(all(model%components_==before),'NCA copy and warm-start independence')
  call require(all(copy%initial_components_==before),'NCA warm starts from prior transformation')
  y=4;model%init='identity';deallocate(model%init_matrix);model%n_components=0
  call model%fit(x,y)
  call require(abs(model%objective_-12)<1.e-12_dp,'NCA one-class objective')
  call require(maxval(abs(model%gradient_))<1.e-12_dp,'NCA one-class gradient vanishes')
  print *, 'NeighborhoodComponentsAnalysis: PASS'
end program
