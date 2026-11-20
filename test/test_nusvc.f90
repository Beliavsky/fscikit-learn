program test_nusvc
  use sklearn, only: dp,nu_svc
  use sklearn_common, only: require
  implicit none
  type(nu_svc) :: model
  real(dp) :: x(24,2),weights(24),mass
  integer :: y(24),i,m,a,b
  real(dp), allocatable :: scores(:,:),before(:,:)
  integer, allocatable :: predicted(:)
  do i=1,24
    x(i,:)=[2*cos(real((i-1)/8,dp)*2.094395102393195_dp)+.1_dp*sin(real(i,dp)), &
      2*sin(real((i-1)/8,dp)*2.094395102393195_dp)+.1_dp*cos(real(i,dp))]
    y(i)=7*((i-1)/8)-5
  end do
  model%nu=.37_dp; model%tol=1.e-9_dp
  call model%fit(x,y)
  call require(model%converged_.and.model%fit_status_==0,'NuSVC multiclass convergence')
  call require(model%score(x,y)==1,'NuSVC separated multiclass fit')
  scores=model%decision_function(x)
  call require(all(shape(scores)==[24,3]),'NuSVC OVR score shape')
  model%break_ties=.true.; predicted=model%predict(x)
  do i=1,24
    call require(predicted(i)==model%classes_(maxloc(scores(i,:),dim=1)),'NuSVC break-ties mapping')
  end do
  model%break_ties=.false.; model%decision_function_shape='ovo'; scores=model%decision_function(x)
  call require(all(shape(scores)==[24,3]),'NuSVC OVO pair count')
  weights=1; weights(1)=0; call model%fit(x,y,weights)
  call require(all(model%pair_alpha_(1,:)==0),'NuSVC excludes zero weight')
  call require(all(model%support_/=1),'NuSVC original sample support indices')
  m=0
  do a=1,2
    do b=a+1,3
      m=m+1
      mass=model%nu*sum(weights,mask=y==model%classes_(a).or.y==model%classes_(b))/2
      call require(abs(sum(model%pair_alpha_(:,m),mask=y==model%classes_(a))-mass)<1.e-10_dp, &
        'NuSVC first class dual mass')
      call require(abs(sum(model%pair_alpha_(:,m),mask=y==model%classes_(b))-mass)<1.e-10_dp, &
        'NuSVC second class dual mass')
    end do
  end do
  model%kernel='linear'; call model%fit(x(:16,:),y(:16))
  call require(size(model%classes_)==2.and.size(model%intercept_)==1,'NuSVC binary refit')
  call require(allocated(model%coef_),'NuSVC linear coefficients')
  scores=model%decision_function(x(:16,:))
  call require(maxval(abs(scores(:,1)-matmul(x(:16,:),model%coef_(1,:))-model%intercept_(1))) &
    <1.e-12_dp,'NuSVC binary sign and primal identity')
  model%kernel='rbf'; model%max_iter=1; call model%fit(x,y)
  call require(.not.allocated(model%coef_),'NuSVC nonlinear refit clears linear coefficients')
  call require(.not.model%converged_.and.model%fit_status_==1,'NuSVC iteration limit diagnostics')
  print *, 'NuSVC tests passed'
end program
