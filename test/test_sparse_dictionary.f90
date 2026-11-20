program test_sparse_dictionary
  use sklearn, only: dp,sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  use sklearn_common, only: require
  implicit none
  type(sparse_pca) :: projection
  type(dictionary_learning) :: learner
  type(sparse_coder) :: coder
  real(dp) :: x(24,5),atoms(3,5)
  real(dp), allocatable :: code(:,:),z(:,:),reconstructed(:,:),again(:,:)
  integer :: i,j
  do i=1,24
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp)),sin(real(4*i,dp))]
  end do
  atoms=0; atoms(1,1)=1; atoms(2,2)=1; atoms(3,3)=1
  code=sparse_encode(x,atoms,algorithm='threshold',alpha=.2_dp)
  do j=1,3
    call require(maxval(abs(code(:,j)-sign(max(abs(x(:,j))-.2_dp,0._dp),x(:,j))))<1.e-14_dp,'threshold coding')
  end do
  coder%dictionary=atoms; coder%transform_algorithm='lasso_cd'; coder%transform_alpha=.2_dp
  again=coder%fit_transform(x)
  call require(maxval(abs(again-code))<1.e-10_dp,'orthogonal Lasso coding agrees with threshold')
  coder%split_sign=.true.; z=coder%transform(x)
  call require(size(z,2)==6.and.all(z>=0),'split-sign features')
  call require(maxval(abs(z(:,:3)-z(:,4:)-code))<1.e-12_dp,'split-sign recombination')
  learner%n_components=3; learner%max_iter=25; learner%alpha=.2_dp
  code=learner%fit_transform(x); z=learner%transform(x)
  call require(size(code,1)==24.and.size(z,2)==3,'dictionary code dimensions')
  call require(all(sum(learner%components_**2,dim=2)<=1+1.e-12_dp),'dictionary atom constraint')
  call require(size(learner%error_)==learner%n_iter_,'dictionary objective history')
  learner%fit_algorithm='cd'; learner%positive_code=.true.; learner%positive_dict=.true.
  learner%transform_algorithm='lasso_cd'; code=learner%fit_transform(abs(x)); z=learner%transform(abs(x))
  call require(all(learner%components_>=0).and.all(code>=0).and.all(z>=0),'positive dictionary/code constraints')
  projection%n_components=3; projection%alpha=.2_dp; projection%max_iter=30
  z=projection%fit_transform(x); reconstructed=projection%inverse_transform(z)
  call require(all(shape(z)==[24,3]).and.all(shape(reconstructed)==shape(x)),'SparsePCA projection dimensions')
  call require(all(abs(sum(projection%components_**2,dim=2)-1)<1.e-10_dp),'SparsePCA normalized components')
  projection%alpha=1.e6_dp; call projection%fit(x)
  call require(all(projection%components_==0),'large penalty clears components')
  z=projection%transform(x); call require(all(z==0),'zero components project to zero')
  reconstructed=projection%inverse_transform(z)
  call require(maxval(abs(reconstructed-spread(projection%mean_,1,24)))<1.e-14_dp,'zero components reconstruct mean')
  projection%max_iter=0; call projection%fit(x)
  call require(projection%n_iter_==0.and.size(projection%error_)==0,'zero-iteration SparsePCA')
  print *, 'Sparse dictionary and SparsePCA tests passed'
end program
