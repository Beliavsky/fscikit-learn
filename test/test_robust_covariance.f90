program test_robust_covariance
  use sklearn, only: dp,min_cov_det,elliptic_envelope,fast_mcd
  use sklearn_common, only: require
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(min_cov_det) :: model
  type(elliptic_envelope) :: envelope
  type(legacy_random_state) :: rng
  real(dp) :: x(80,2),one(7,1),query(2,2),identity(2,2),comp(2,2)
  real(dp), allocatable :: location(:),covariance(:,:),dist(:),precision(:,:)
  logical, allocatable :: support(:)
  integer, allocatable :: labels(:)
  integer :: i
  one(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp,100._dp]
  call fast_mcd(one,location,covariance,support,dist)
  call require(abs(location(1)-2.5_dp)<1.e-14_dp.and.abs(covariance(1,1)-2)<1.e-14_dp,'univariate shortest-half fit')
  call model%fit(one)
  call require(count(model%support_)==6,'univariate reweighting excludes distant observation')
  call require(abs(model%location_(1)-2.5_dp)<1.e-14_dp,'univariate reweighted location')
  call require(abs(model%covariance_(1,1)-35._dp/12)<1.e-14_dp,'univariate reweighted covariance')
  call rng%seed(2026)
  do i=1,80
    x(i,:)=[rng%normal(),rng%normal()]; x(i,2)=x(i,2)+.3_dp*x(i,1)
  end do
  x(:10,:)=x(:10,:)+10; model%random_state=42; call model%fit(x)
  call require(count(model%raw_support_)==42,'MCD automatic raw support count')
  call require(maxval(abs(model%location_))<1,'robust center resists contamination')
  precision=model%get_precision(); identity=matmul(model%covariance_,precision)
  call require(maxval(abs(identity-reshape([1._dp,0._dp,0._dp,1._dp],[2,2])))<1.e-10_dp,'MCD precision inverse')
  comp=model%covariance_; comp(1,1)=comp(1,1)+.2_dp
  call require(abs(model%error_norm(comp)-.02_dp)<1.e-12_dp,'MCD Frobenius error')
  call require(abs(model%error_norm(comp,norm='spectral')-.02_dp)<1.e-12_dp,'MCD spectral error')
  model%store_precision=.false.; call model%fit(x)
  call require(.not.allocated(model%precision_).and.model%rank_==2,'MCD uncached full-rank precision')
  precision=model%get_precision()
  envelope%random_state=42; labels=envelope%fit_predict(x)
  call require(count(labels==-1)==8,'elliptic empirical contamination')
  call require(envelope%score(x,labels)==1,'elliptic accuracy')
  query(1,:)=0; query(2,:)=20
  labels=envelope%predict(query)
  call require(all(labels==[1,-1]),'elliptic center and distant labels')
  model%assume_centered=.true.; call model%fit(x)
  call require(all(model%location_==0).and.all(model%raw_location_==0),'MCD assumed centering')
  print *, 'Robust covariance tests passed'
end program
