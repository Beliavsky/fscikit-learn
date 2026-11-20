program nonlinear_feature_scoring
  use sklearn, only: dp,mutual_info_classif,mutual_info_regression,r_regression, &
    mutual_info_score,generic_univariate_select
  implicit none
  integer, parameter :: n=80
  real(dp) :: x(n,4),y(n),t
  integer :: i,labels(n)
  real(dp), allocatable :: information(:),correlation(:),class_information(:),selected(:,:)
  type(generic_univariate_select) :: selector
  do i=1,n
    t=4._dp*(i-1)/(n-1)-2
    x(i,:)=[t,sin(2.3_dp*i),real(mod(i,3),dp),cos(1.7_dp*i)]
    y(i)=t*t+.1_dp*x(i,3)+.03_dp*sin(real(i,dp))
  end do
  information=mutual_info_regression(x,y,discrete_features=[3],random_state=42)
  correlation=r_regression(x,y)
  selector%mode='k_best'; selector%param=2
  call selector%fit_scores(information); selected=selector%transform(x)
  labels=merge(1,0,y>sum(y)/n)
  class_information=mutual_info_classif(x,labels,discrete_features=[3],random_state=42)
  print *, 'Regression mutual information (nats):',information
  print *, 'Signed Pearson correlations:',correlation
  print *, 'Selected feature indices:',selector%get_support_indices()
  print *, 'Selected matrix shape:',shape(selected)
  print *, 'Classification mutual information:',class_information
  print *, 'Target partition self-information:',mutual_info_score(labels,labels)
end program
