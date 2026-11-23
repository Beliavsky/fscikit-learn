program test_mlp_early
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(mlp_classifier) :: c,first
  type(mlp_regressor) :: r,initial
  real(dp) :: x(20,2),y(20),score
  integer :: i,labels(20)
  do i=1,20
    x(i,:)=[real(i-10,dp)/10,real(mod(i,3),dp)/10]
    labels(i)=mod(i,2);y(i)=sin(x(i,1))
  end do
  c%hidden_layer_sizes=[4];c%early_stopping=.true.;c%validation_fraction=.3_dp
  c%solver='sgd';c%learning_rate_init=.01_dp;c%tol=2;c%n_iter_no_change=1;c%max_iter=20
  first=c;first%max_iter=1
  call first%fit(x,labels);call c%fit(x,labels)
  call require(c%n_iter_==3.and.size(c%validation_scores_)==3,'validation patience uses strict greater-than limit')
  call require(c%t_==42,'early stopping counts only training partition')
  call require(all(c%training_indices_==first%training_indices_).and. &
    all(c%validation_indices_==first%validation_indices_),'split is independent of iteration limit')
  call require(count(labels(c%validation_indices_)==0)==3.and.count(labels(c%validation_indices_)==1)==3, &
    'binary validation partition is stratified')
  call require(ieee_is_nan(c%best_loss_),'early stopping has no best training loss')
  do i=1,size(c%layers_)
    call require(all(c%layers_(i)%weights==c%best_layers_(i)%weights).and. &
      all(c%layers_(i)%bias==c%best_layers_(i)%bias),'fit restores best validation parameters')
  end do
  score=c%score(x(c%validation_indices_,:),labels(c%validation_indices_))
  call require(score==c%best_validation_score_,'restored classifier attains saved validation score')
  c%warm_start=.true.;c%max_iter=2;call c%fit(x,labels)
  call require(size(c%validation_scores_)==c%n_iter_+3,'warm fit appends validation history')
  r%hidden_layer_sizes=[integer::];r%early_stopping=.true.;r%validation_fraction=.01_dp
  r%max_iter=4;initial=r;initial%max_iter=1
  call initial%fit(x,y);call r%fit(x,y)
  call require(all(ieee_is_nan(r%validation_scores_)).and.r%no_improvement_count_==0, &
    'one-row R2 validation is NaN without consuming patience')
  call require(all(r%layers_(1)%weights==initial%layers_(1)%weights).and. &
    all(r%layers_(1)%bias==initial%layers_(1)%bias),'NaN validation restores initialization')
  r%warm_start=.true.;call r%fit(x,y)
  call require(any(r%layers_(1)%bias/=initial%layers_(1)%bias),'warm restored checkpoint follows source alias updates')
  print *, 'MLP validation splits, histories and best checkpoints: PASS'
end program
