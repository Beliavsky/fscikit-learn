program test_density
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(kernel_density) :: kde,weighted
  real(dp) :: center(1,1),query(10001,1),integral,mean,variance,training(2,1),probe(1,1)
  real(dp), allocatable :: scores(:),samples(:,:),repeat(:,:)
  character(16) :: kernels(5)=[character(16)::'gaussian','tophat','epanechnikov','exponential','linear']
  integer :: i,k
  center=0; kde%bandwidth=0.7_dp
  do i=1,10001
    query(i,1)=-10+(i-1)*0.002_dp
  end do
  do k=1,5
    kde%kernel=kernels(k); call kde%fit(center); scores=kde%score_samples(query)
    integral=sum(exp(scores))*0.002_dp
    call require(abs(integral-1)<0.003_dp,'density kernel integrates to one')
  end do
  probe=1000; kde%kernel='linear'; call kde%fit(center); scores=kde%score_samples(probe)
  call require(.not.ieee_is_finite(scores(1)).and.scores(1)<0,'compact kernel returns negative infinity')
  kde%kernel='gaussian'; call kde%fit(center); scores=kde%score_samples(probe)
  call require(ieee_is_finite(scores(1)),'Gaussian log density avoids probability underflow')
  training(:,1)=[0._dp,4._dp]; weighted%bandwidth=0.7_dp
  call weighted%fit(training,sample_weight=[0._dp,1._dp]); probe=4; center=4; call kde%fit(center)
  call require(maxval(abs(weighted%score_samples(probe)-kde%score_samples(probe)))<1.e-12_dp, &
    'density zero sample weight')
  samples=weighted%sample(6000,random_state=17); repeat=weighted%sample(6000,random_state=17)
  call require(all(samples==repeat),'density sampling reproducibility')
  mean=sum(samples)/6000; variance=sum((samples-mean)**2)/6000
  call require(abs(mean-4)<0.06_dp.and.abs(variance-0.49_dp)<0.04_dp,'weighted Gaussian sampling moments')
  weighted%kernel='tophat'; call weighted%fit(training,sample_weight=[0._dp,1._dp])
  samples=weighted%sample(6000,random_state=17); mean=sum(samples)/6000; variance=sum((samples-mean)**2)/6000
  call require(all(abs(samples-4)<=0.7_dp).and.abs(variance-0.49_dp/3)<0.02_dp,'uniform ball sampling')
  weighted%bandwidth_rule='scott'; call weighted%fit(training)
  call require(abs(weighted%bandwidth_-2._dp**(-0.2_dp))<1.e-14_dp,'Scott bandwidth')
  weighted%bandwidth_rule='silverman'; call weighted%fit(training)
  call require(abs(weighted%bandwidth_-1.5_dp**(-0.2_dp))<1.e-14_dp,'Silverman bandwidth')
  print *, 'kernel density: PASS'
end program
