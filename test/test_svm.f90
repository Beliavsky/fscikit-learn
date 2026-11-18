program test_svm
  use sklearn_common
  use sklearn_svm
  implicit none
  type(svc) :: svm
  real(dp) :: x(8,2),xor(4,2)
  integer :: y(8)
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]; x(:,2)=0
  y=[-3,-3,-3,-3,5,5,5,5]
  svm%kernel='linear'; call svm%fit(x,y)
  call require(svm%score(x,y)==1.and.svm%converged_,'linear SVM')
  xor(:,1)=[0._dp,0._dp,1._dp,1._dp]; xor(:,2)=[0._dp,1._dp,0._dp,1._dp]
  svm%kernel='rbf'; svm%c=100; svm%gamma=2
  call svm%fit(xor,[0,1,1,0]); call require(svm%score(xor,[0,1,1,0])==1,'RBF XOR')
  y=[0,0,0,1,1,2,2,2]; svm%kernel='linear'; call svm%fit(x,y)
  call require(svm%score(x,y)==1,'multiclass one-versus-one SVM')
  print *, 'support vector classification: PASS'
end program
