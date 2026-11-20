program statistical_feature_selection
  use sklearn, only: dp,select_fdr,linear_regression
  implicit none
  type(select_fdr) :: selector
  type(linear_regression) :: model
  real(dp) :: x(100,8),y(100)
  real(dp), allocatable :: selected(:,:),restored(:,:)
  integer, allocatable :: indices(:)
  integer :: i,j
  do i=1,100
    do j=1,8
      x(i,j)=sin(real(i*j,dp)*.31_dp)+cos(real(i*(j+1),dp)*.17_dp)
    end do
  end do
  y=1+2*x(:,1)-1.5_dp*x(:,3)+.1_dp*sin([(real(i,dp),i=1,100)])
  selector%alpha=.05_dp; call selector%fit(x,y)
  indices=selector%get_support_indices(); selected=selector%transform(x)
  restored=selector%inverse_transform(selected)
  call model%fit(selected,y)
  print *, 'Selected feature indices (one-based): ',indices
  print *, 'Selected p-values: ',selector%pvalues_(indices)
  print *, 'Selected/restored column counts: ',size(selected,2),size(restored,2)
  print *, 'Training R2 after feature selection: ',model%score(selected,y)
end program
