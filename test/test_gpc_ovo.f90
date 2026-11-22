program test_gpc_ovo
  use sklearn_common, only: dp,require
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_gp_kernels
  implicit none
  type(gaussian_process_classifier) :: model,binary
  real(dp) :: x(15,1)
  real(dp), allocatable :: first(:),probability(:,:)
  integer :: y(15),i,k,n
  integer, allocatable :: predicted(:),before(:)
  x(:,1)=[(real(i,dp)/5-1.6_dp,i=1,15)]
  y=[-4,-4,-4,-4,9,9,9,9,9,17,17,17,17,17,17]
  model%multi_class='one_vs_one';model%kernel=rbf_covariance(.7_dp)+white_covariance(.04_dp)
  call model%fit(x,y)
  call require(all(model%n_training_samples_==[9,10,11]),'GPC OVO imbalanced pair subset sizes')
  call require(all(reshape(model%class_pairs_,[6])==[1,2,1,3,2,3]),'GPC OVO lexicographic pairs')
  before=model%predict(x);call require(all(before==y),'GPC OVO fitted training predictions')
  do k=1,3
    n=model%n_training_samples_(k)
    call require(all(model%pi_(n+1:,k)==0),'GPC OVO padded state is zero')
    call require(all(model%training_indices_(:n,k)>0),'GPC OVO training indices are one based')
  end do
  first=model%kernels_(1)%theta()
  call require(abs(model%log_marginal_likelihood(first)-model%log_marginal_likelihood())<1.e-12_dp, &
    'GPC OVO shared theta likelihood')
  call model%fit(x,y);predicted=model%predict(x)
  call require(all(predicted==before),'GPC OVO refit resets subset state')
  model%multi_class='one_vs_rest';call model%fit(x,y)
  call require(all(model%n_training_samples_==15),'GPC OVR refit clears pair subsets')
  probability=model%predict_proba(x)
  call require(maxval(abs(sum(probability,dim=2)-1))<1.e-14_dp,'GPC OVR probabilities after OVO refit')
  y=merge(-4,9,x(:,1)<0);model%multi_class='one_vs_one';binary%kernel=model%kernel
  call model%fit(x,y);call binary%fit(x,y)
  probability=model%predict_proba(x)
  call require(maxval(abs(probability-binary%predict_proba(x)))<1.e-14_dp,'GPC binary OVO preserves probabilities')
  call require(all(model%predict(x)==binary%predict(x)),'GPC binary modes agree')
  print *, 'Gaussian process one-vs-one classification: PASS'
end program
