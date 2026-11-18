program test_kernel_approximation
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(rbf_sampler) :: rbf,repeat
  type(nystroem) :: ny
  real(dp) :: x(12,3),small(4,2),mean,variance
  real(dp), allocatable :: z(:,:),again(:,:),gram(:,:),actual(:,:),weights(:,:)
  integer :: i
  do i=1,12
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),real(i,dp)/12]
  end do
  rbf%n_components=20000; rbf%gamma=0.7_dp; rbf%random_state=42
  z=rbf%fit_transform(x); gram=rbf_kernel(x,x,0.7_dp); actual=matmul(z,transpose(z))
  call require(maxval(abs(actual-gram))<0.04_dp,'random Fourier features approximate RBF kernel')
  mean=sum(rbf%random_weights_)/size(rbf%random_weights_)
  variance=sum(rbf%random_weights_**2)/size(rbf%random_weights_)-mean**2
  call require(abs(mean)<0.03_dp.and.abs(variance-1.4_dp)<0.05_dp,'RBF Fourier normal distribution')
  call require(all(rbf%random_offset_>=0).and.all(rbf%random_offset_<2*acos(-1._dp)), &
    'RBF offsets in uniform phase range')
  call require(abs(sum(cos(rbf%random_offset_))/rbf%n_components)<0.03_dp,'RBF uniform phase moment')
  repeat=rbf; again=repeat%fit_transform(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'RBF seeded repeatability')
  rbf%gamma_scale=.true.; z=rbf%fit_transform(x)
  mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x)
  call require(abs(rbf%gamma_-1/(3*variance))<1.e-12_dp,'RBF scale uses global feature variance')
  small=1; call rbf%fit(small); call require(rbf%gamma_==1,'RBF scale constant fallback')
  ny%n_components=20; ny%gamma=0.7_dp; ny%random_state=42
  z=ny%fit_transform(x); actual=matmul(z,transpose(z))
  call require(size(z,2)==12,'Nystroem caps landmarks at sample count')
  call require(maxval(abs(actual-gram))<1.e-9_dp,'full RBF landmarks reconstruct exact kernel')
  weights=ny%components_; again=ny%fit_transform(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'Nystroem seeded repeatability')
  do i=1,12
    call require(count(ny%component_indices_==i)==1,'Nystroem landmarks sampled without replacement')
  end do
  ny%kernel='linear'; z=ny%fit_transform(x); gram=linear_kernel(x,x)
  call require(maxval(abs(matmul(z,transpose(z))-gram))<1.e-7_dp,'Nystroem singular linear basis')
  small(:,1)=[0._dp,1._dp,2._dp,3._dp]; small(:,2)=small(:,1)
  ny%kernel='cosine'; z=ny%fit_transform(small); gram=cosine_similarity(small,small)
  call require(maxval(abs(matmul(z,transpose(z))-gram))<1.e-7_dp,'Nystroem duplicate and zero vectors')
  print *, 'kernel approximations: PASS'
end program
