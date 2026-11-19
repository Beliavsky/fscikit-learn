program passive_aggressive
  use sklearn
  implicit none
  type(passive_aggressive_classifier) :: classifier
  type(passive_aggressive_regressor) :: reg
  real(dp) :: x(40,2),y(40),t
  integer :: labels(40),i
  do i=1,40
    t=real(i,dp)/40; x(i,:)=[sin(17*t),cos(23*t)]
    y(i)=2*x(i,1)-0.7_dp*x(i,2)+0.3_dp; labels(i)=merge(1,0,x(i,1)>0)
  end do
  classifier%shuffle=.false.; reg%shuffle=.false.; reg%epsilon=0.01_dp; reg%c=0.1_dp
  call classifier%fit(x,labels); call reg%fit(x,y)
  print *, 'Training accuracy:',classifier%score(x,labels)
  print *, 'Regression R2:',reg%score(x,y)
  print *, 'Epochs:',classifier%n_iter_,reg%n_iter_
end program
