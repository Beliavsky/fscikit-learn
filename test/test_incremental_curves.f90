program test_incremental_curves
  use sklearn_validation
  use sklearn_common, only: dp,require
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_naive_bayes, only: gaussian_nb
  implicit none
  type(sgd_regressor) :: base,fresh
  type(sgd_classifier) :: classifier
  type(gaussian_nb) :: bayes
  type(learning_curve_options) :: options
  type(learning_curve_result) :: result
  real(dp) :: x(36,3),y(36),score
  integer :: i,j,previous,labels(36)
  integer, allocatable :: ids(:),chunk(:)
  do i=1,36
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp))]
  end do
  y=2*x(:,1)-x(:,2);labels=merge(9,-4,y>0)
  base%learning_rate='constant';base%eta0=.02_dp;base%tol=-1;base%max_iter=3
  fresh=base;call base%fit(x,y)
  options%cv=3;options%train_counts=[4,12,24];options%exploit_incremental_learning=.true.;options%return_times=.true.
  result=learning_curve(base,x,y,options)
  do i=1,3
    fresh=sgd_regressor();fresh%learning_rate='constant';fresh%eta0=.02_dp;fresh%tol=-1;previous=0
    do j=1,3
      ids=result%folds(i)%train(:result%train_sizes(j));chunk=ids(previous+1:)
      call fresh%partial_fit(x(chunk,:),y(chunk))
      score=fresh%score(x(ids,:),y(ids))
      call require(abs(score-result%train_score(j,i))<1.e-12_dp,'incremental regression retains state across chunks')
      previous=size(ids)
    end do
  end do
  classifier%loss='log_loss';classifier%learning_rate='constant';classifier%eta0=.02_dp;classifier%tol=-1
  result=learning_curve(classifier,x,labels,options)
  call require(all(result%test_score>=0).and.all(result%test_score<=1),'incremental classifier scores')
  result=learning_curve(bayes,x,labels,options)
  call require(all(result%test_score>=0).and.all(result%test_score<=1),'incremental Gaussian NB scores')
  call require(all(result%actual_train_sizes(:,1)==[4,12,24]),'incremental training prefix sizes')
  print *, 'Incremental learning curves: PASS'
end program
