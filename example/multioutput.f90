program multioutput
  use sklearn, only: dp,multi_output_regressor,multi_output_classifier,linear_regression,gaussian_nb, &
    multioutput_probability
  implicit none
  type(multi_output_regressor) :: reg
  type(multi_output_classifier) :: clf
  type(linear_regression) :: linear
  type(gaussian_nb) :: bayes
  type(multioutput_probability), allocatable :: probability(:)
  real(dp) :: x(8,1),y(8,2),query(2,1)
  integer :: labels(8,2),i
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]
  y(:,1)=2*x(:,1)+1; y(:,2)=-x(:,1)+3; query(:,1)=[-2._dp,2._dp]
  call reg%fit(x,y,linear)
  print *, 'Multioutput regression predictions:',reg%predict(query)
  labels(:,1)=[0,0,0,0,1,1,1,1]; labels(:,2)=[-7,-7,4,4,11,11,11,11]
  call clf%fit(x,labels,bayes)
  probability=clf%predict_proba(query)
  print *, 'Multioutput classifier predictions:',clf%predict(query)
  do i=1,size(probability)
    print *, 'Output',i,'classes:',probability(i)%classes
  end do
end program
