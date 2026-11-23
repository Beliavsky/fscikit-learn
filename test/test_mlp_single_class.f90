program test_mlp_single_class
  use sklearn_neural_network, only: mlp_classifier
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_common, only: dp,require
  implicit none
  type(mlp_classifier) :: c,stream,early
  type(csr_matrix) :: sparse
  real(dp) :: x(8,2)
  real(dp),allocatable :: prob(:,:)
  integer :: i
  do i=1,8
    x(i,:)=[real(i,dp)/8,real(mod(i,3),dp)/4]
  end do
  sparse=csr_from_dense(x);c%hidden_layer_sizes=[integer::];c%max_iter=1
  call c%fit(x,spread(-11,1,8))
  call require(c%single_class_.and.size(c%classes_)==1.and.c%n_outputs_==1,'single-class logistic layout')
  c%layers_(1)%weights=0;c%layers_(1)%bias=10
  prob=c%predict_proba(x)
  call require(size(prob,2)==2.and.all(prob(:,2)>.5_dp),'source single-class probability expansion')
  call require(all(c%predict(x)==-11).and.all(c%predict(sparse)==-11),'single-class prediction ignores sigmoid threshold')
  call require(c%score(sparse,spread(-11,1,8))==1,'single-class score uses the sole label')
  call require(c%score(sparse,spread(7,1,8))==0,'single-class query score can use unseen labels')
  stream%hidden_layer_sizes=[integer::]
  call stream%partial_fit(sparse,reshape(spread(7,1,8),[8,1]),classes=[7,7])
  call stream%partial_fit(x,spread(7,1,8))
  call require(stream%single_class_.and.stream%t_==16.and.all(stream%predict(sparse)==7), &
    'single declared class persists across storage and target-rank changes')
  early%hidden_layer_sizes=[integer::];early%early_stopping=.true.;early%validation_fraction=.5_dp
  early%max_iter=20;early%tol=1.e-4_dp;early%n_iter_no_change=1
  call early%fit(sparse,spread(0,1,8))
  call require(early%n_iter_==3.and.all(early%validation_scores_==1),'single-class validation accuracy and patience')
  call require(early%best_validation_score_==1.and.early%t_==12,'single-class holdout clock and best score')
  early%warm_start=.true.;early%tol=0;early%max_iter=4;call early%fit(x,spread(0,1,8))
  call require(early%n_iter_==4.and.size(early%validation_scores_)==7,'zero-tolerance flat validation does not stop')
  early%warm_start=.false.;early%early_stopping=.false.;call early%fit(x,[0,1,0,1,0,1,0,1])
  call require(.not.early%single_class_.and.size(early%classes_)==2,'cold fit can expand the class set')
  call early%fit(sparse,reshape(spread(7,1,8),[8,1]))
  call require(early%single_class_.and.all(early%predict(x)==7),'cold one-column target can collapse the class set')
  print *, 'MLP single-class labels, probabilities, validation and streaming: PASS'
end program
