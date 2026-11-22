program test_sample_generators
  use sklearn, only: dp,make_friedman1,make_friedman2,make_friedman3,make_sparse_uncorrelated, &
    make_s_curve,make_swiss_roll,make_gaussian_quantiles
  use sklearn_common, only: require
  implicit none
  real(dp), allocatable :: x(:,:),y(:),a(:,:),b(:)
  integer, allocatable :: labels(:)
  real(dp) :: pi
  integer :: i
  pi=acos(-1._dp)
  call make_friedman1(17,9,x,y)
  call make_friedman1(17,9,a,b,random_state=0)
  call require(all(x==a).and.all(y==b),'sample generators default to deterministic seed zero')
  call require(maxval(abs(y-(10*sin(pi*x(:,1)*x(:,2))+20*(x(:,3)-.5_dp)**2+10*x(:,4)+5*x(:,5)))) &
    <1.e-12_dp,'Friedman1 noiseless formula')
  call make_friedman2(17,x,y,random_state=42)
  call require(maxval(abs(y-sqrt(x(:,1)**2+(x(:,2)*x(:,3)-1/(x(:,2)*x(:,4)))**2))) &
    <1.e-12_dp,'Friedman2 noiseless formula')
  call make_friedman3(17,a,b,random_state=42)
  call require(all(a==x),'Friedman2 and Friedman3 share feature sampling')
  call require(maxval(abs(b-atan((a(:,2)*a(:,3)-1/(a(:,2)*a(:,4)))/a(:,1)))) &
    <1.e-12_dp,'Friedman3 noiseless formula')
  call make_sparse_uncorrelated(17,4,x,y,random_state=42)
  call require(all(shape(x)==[17,4]).and.size(y)==17,'sparse uncorrelated dense output')
  call make_s_curve(17,x,y,random_state=42)
  call require(maxval(abs(x(:,1)-sin(y)))<1.e-14_dp,'S curve latent coordinate')
  call make_swiss_roll(17,x,y,random_state=42,hole=.true.)
  call require(maxval(abs(x(:,1)**2+x(:,3)**2-y**2))<1.e-12_dp,'Swiss roll radial coordinate')
  call require(.not.any(y>2.5_dp*pi.and.y<3.5_dp*pi.and.x(:,2)>7.and.x(:,2)<14),'Swiss roll hole')
  call make_gaussian_quantiles(17,3,x,labels,n_classes=4,shuffle=.false.,random_state=42)
  call require(all(labels==[(min((i-1)/4,3),i=1,17)]),'Gaussian quantile class allocation')
  call require(all(sum(x(2:,:)**2,dim=2)>=sum(x(:16,:)**2,dim=2)),'Gaussian quantile radius sorting')
  call make_gaussian_quantiles(17,3,x,labels,n_classes=4,cov=0._dp,mean=[1._dp,2._dp,3._dp])
  call require(all(x(:,1)==1).and.all(x(:,2)==2).and.all(x(:,3)==3),'zero covariance Gaussian quantiles')
  print *, 'Seeded sample generators: PASS'
end program
