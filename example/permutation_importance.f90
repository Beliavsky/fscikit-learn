program inspect_features
  use sklearn, only: dp,linear_regression,permutation_importance,permutation_importance_result
  implicit none
  type(linear_regression) :: model
  type(permutation_importance_result) :: importance
  real(dp) :: x(20,2),y(20)
  integer :: i
  x(:,1)=[(real(i,dp),i=1,20)]; x(:,2)=[(real(mod(i,3),dp),i=1,20)]
  y=2*x(:,1)
  call model%fit(x,y)
  importance=permutation_importance(x,score,random_state=42)
  print *, 'Mean importance:',importance%importances_mean
  print *, 'Importance deviation:',importance%importances_std
contains
  function score(features,rows,sample_weight) result(value)
    real(dp), intent(in) :: features(:,:)
    integer, intent(in) :: rows(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: value
    value=model%score(features,y(rows))
  end function
end program
