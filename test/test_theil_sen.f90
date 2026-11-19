program test_theil_sen
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(theil_sen_regressor) :: robust,repeat,capped,scalar,wide
  type(linear_regression) :: ols
  real(dp) :: x(30,2),y(30),clean(30),sx(6,1),sy(6),wx(2,3),wy(2)
  integer :: i,j,k
  do i=1,30
    x(i,1)=real(i,dp)/10; x(i,2)=2*x(i,1)
  end do
  clean=2*x(:,1)+0.5_dp; y=clean; y(:4)=y(:4)+20
  robust%tol=1.e-9_dp; robust%max_iter=2000
  call robust%fit(x(:,1:1),y); call ols%fit(x(:,1:1),y)
  call require(robust%converged_.and.robust%exhaustive_,'exhaustive robust convergence')
  call require(robust%n_subsamples_==2.and.robust%n_subpopulation_==435,'subset count and resolved size')
  call require(all(robust%subsets_(1,:)==[1,2]).and.all(robust%subsets_(435,:)==[29,30]), &
    'lexicographic subset enumeration')
  call require(abs(robust%coef_(1)-2)<1.e-6_dp.and.abs(robust%intercept_-0.5_dp)<1.e-6_dp, &
    'TheilSen recovers contaminated line')
  call require(sum((robust%predict(x(:,1:1))-clean)**2)<sum((ols%predict(x(:,1:1))-clean)**2)*1.e-8_dp, &
    'TheilSen improves contaminated least squares')
  call require(robust%breakdown_>0.28_dp.and.robust%breakdown_<0.3_dp,'pairwise breakdown estimate')
  robust%n_subsamples=30; call robust%fit(x(:,1:1),y)
  call require(robust%n_subpopulation_==1,'full subset population')
  call require(abs(robust%breakdown_-(1-0.5_dp**(1._dp/30))/30)<1.e-15_dp, &
    'finite-sample approximate breakdown')
  call require(abs(robust%coef_(1)-ols%coef_(1))<1.e-10_dp,'full subset equals least squares')
  robust%n_subsamples=0; robust%max_subpopulation=17; robust%random_state=19
  call robust%fit(x(:,1:1),y); repeat=robust; call repeat%fit(x(:,1:1),y)
  call require(.not.robust%exhaustive_.and.robust%n_subpopulation_==17,'sampled subset cap')
  call require(all(robust%subsets_==repeat%subsets_).and.all(robust%coef_==repeat%coef_), &
    'sampled subset reproducibility')
  do i=1,robust%n_subpopulation_
    call require(all(robust%subsets_(i,:)>=1).and.all(robust%subsets_(i,:)<=30),'valid subset indices')
    do j=1,robust%n_subsamples_
      do k=j+1,robust%n_subsamples_
        call require(robust%subsets_(i,j)/=robust%subsets_(i,k),'sampling without replacement within subset')
      end do
    end do
  end do
  capped%max_iter=1; capped%tol=1.e-14_dp; call capped%fit(x(:,1:1),y)
  call require(.not.capped%converged_.and.capped%n_iter_==0,'capped spatial median status and index')
  do i=1,6
    sx(i,1)=i; sy(i)=real(i*i,dp)
  end do
  scalar%fit_intercept=.false.; call scalar%fit(sx,sy)
  call require(scalar%coef_(1)==3.5_dp.and.scalar%intercept_==0,'scalar coefficient median')
  call require(scalar%n_iter_==1.and.scalar%converged_,'scalar median shortcut')
  robust%max_subpopulation=10000; call robust%fit(x,3*x(:,1)+0.5_dp)
  call require(maxval(abs(robust%coef_-[0.6_dp,1.2_dp]))<1.e-8_dp,'rank-deficient minimum-norm subsets')
  wx=0; wx(1,1)=1; wx(2,2)=1; wy=[2._dp,4._dp]
  wide%fit_intercept=.false.; call wide%fit(wx,wy)
  call require(all(abs(wide%coef_-[2._dp,4._dp,0._dp])<1.e-12_dp),'underdetermined minimum-norm fit')
  call require(wide%n_subsamples_==2.and.wide%n_subpopulation_==1,'underdetermined uses all rows')
  print *, 'TheilSenRegressor tests passed'
end program
