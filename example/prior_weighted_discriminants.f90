program prior_weighted_discriminants
  use sklearn_common, only: dp
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis,quadratic_discriminant_analysis
  implicit none
  type(linear_discriminant_analysis) :: linear
  type(quadratic_discriminant_analysis) :: quadratic
  real(dp) :: x(30,3)
  real(dp), allocatable :: probability(:,:)
  integer :: labels(30),i,j
  do i=1,30
    labels(i)=mod(i,3)
    do j=1,3
      x(i,j)=sin(.37_dp*i*j)+labels(i)*.4_dp*j
    end do
  end do
  linear%priors=[0._dp,2._dp,6._dp];linear%shrinkage_mode='auto';call linear%fit(x,labels)
  probability=linear%predict_proba(x(:1,:))
  print '(a,3f14.8)', 'LDA fitted priors: ',linear%priors_
  print '(a,3f14.8)', 'First-row probabilities: ',probability(1,:)
  quadratic%priors=[1._dp,3._dp,6._dp];quadratic%reg_param=.2_dp;call quadratic%fit(x,labels)
  print '(a,3f14.8)', 'QDA fitted prior weights: ',quadratic%priors_
end program
