program test_regressor_chain
  use sklearn, only: dp,regressor_chain,linear_regression,ridge,sgd_regressor,index_split
  use sklearn_common, only: require
  implicit none
  type(regressor_chain) :: chain,saved
  type(linear_regression) :: base
  type(ridge) :: regularized
  type(index_split) :: splits(2)
  real(dp) :: x(8,1),y(8,2),w(8)=1
  real(dp), allocatable :: prediction(:,:)
  integer :: j
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]
  y(:,1)=2*x(:,1)+1; y(:,2)=-3*x(:,1)+4
  call base%fit(x,0*y(:,1)); chain%order=[2,1]
  call chain%fit(x,y,base); prediction=chain%predict(x)
  call require(maxval(abs(prediction-y))<1.e-11_dp,'chain restores original output order')
  call require(all(chain%order_==[2,1]),'explicit one-based chain order')
  call require(all(base%coef_==0),'chain preserves pretrained template')
  do j=1,2
    select type(model=>chain%estimators_(j)%estimator_)
    class is(linear_regression)
      call require(size(model%coef_)==j,'chain augments base input width')
    end select
  end do
  call require(abs(chain%score(x,y,w)-1)<1.e-14_dp,'chain weighted uniform-average R2')
  saved=chain; regularized%alpha=.7_dp; chain%cv=2
  call chain%fit(x,y,regularized); prediction=chain%predict(x)
  call require(all(shape(prediction)==[8,2]),'cross-validated chain dimensions')
  call require(maxval(abs(saved%predict(x)-y))<1.e-11_dp,'chain deep copy preserves fitted state')
  splits(1)%train=[5,6,7,8]; splits(1)%test=[1,2,3,4]
  splits(2)%train=[1,2,3,4]; splits(2)%test=[5,6,7,8]
  chain%cv=0; call chain%fit(x,y,regularized,splits=splits)
  call require(maxval(abs(chain%predict(x)-prediction))<1.e-13_dp,'explicit folds match integer KFold')
  deallocate(chain%order); chain%random_order=.true.; chain%random_state=0
  call chain%fit(x,y,base)
  call require(all(chain%order_==[2,1]),'legacy random chain permutation')
  call chain%fit(x,y(:,:1),base)
  call require(chain%n_outputs_==1.and.size(chain%estimators_)==1,'chain refit replaces output state')
  print *, 'regressor chains: PASS'
end program
