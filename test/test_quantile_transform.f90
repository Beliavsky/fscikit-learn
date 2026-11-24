program test_quantile_transform
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan,ieee_is_finite
  implicit none
  type(quantile_transformer) :: model,again
  real(dp) :: x(4,1),q(5,1),missing(6,3),large(40,2),ties(9,1)
  real(dp), allocatable :: z(:,:),back(:,:),first(:,:)
  integer :: i
  x(:,1)=[0._dp,1._dp,1._dp,2._dp]
  model%n_quantiles=4; model%subsample=0
  z=model%fit_transform(x)
  call require(maxval(abs(z(:,1)-[0._dp,2._dp/3,2._dp/3,1._dp]))<1.e-14_dp, &
    'source percentile-grid rounding near repeated observations')
  back=model%inverse_transform(z)
  call require(maxval(abs(back-x))<1.e-14_dp,'uniform inverse')
  q(:,1)=[-1._dp,0._dp,1._dp,2._dp,3._dp]
  z=model%transform(q)
  call require(all(z(:,1)==[0._dp,0._dp,2._dp/3,1._dp,1._dp]),'uniform extrapolation clips')
  model%output_distribution='normal'; z=model%transform(x)
  call require(all(ieee_is_finite(z)).and.abs(z(1,1)+5.1993375826_dp)<1.e-9_dp,'finite normal tail clipping')
  call require(abs(z(2,1)-.43072729929545744_dp)<1.e-14_dp.and.z(2,1)==z(3,1), &
    'normal scores retain percentile-grid rounding')
  back=model%inverse_transform(z)
  call require(maxval(abs(back-x))<1.e-13_dp,'normal inverse restores clipped endpoints')
  missing(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]; missing(:,2)=2
  missing(:,3)=ieee_value(0._dp,ieee_quiet_nan); missing(2,1)=missing(1,3)
  model%n_quantiles=20; z=model%fit_transform(missing)
  call require(model%n_quantiles_==6,'quantile count capped at sample count')
  call require(ieee_is_nan(z(2,1)).and.all(ieee_is_nan(z(:,3))),'NaNs ignored in fit and preserved in transform')
  call require(all(ieee_is_nan(model%quantiles_(:,3))),'all-missing quantile column')
  call require(all(z(:,2)<-5),'constant normal column maps to lower bound')
  model%output_distribution='uniform'; model%n_quantiles=1; z=model%fit_transform(x)
  call require(all(z==0).and.model%references_(1)==0,'single quantile table')
  do i=1,40
    large(i,:)=[real(i,dp),real(i*i,dp)]
  end do
  model%n_quantiles=5; model%subsample=10; model%random_state=23
  call model%fit(large); first=model%quantiles_; again=model
  call again%fit(large)
  call require(all(first==again%quantiles_),'native subsampling reproducible')
  call require(all(model%quantiles_(2:,:)>=model%quantiles_(:4,:)),'subsample quantiles monotonic')
  call require(all(model%quantiles_(:,1)>=1).and.all(model%quantiles_(:,1)<=40),'subsample lies inside data range')
  ties(:,1)=[0._dp,0._dp,.5_dp,1._dp,1._dp,1._dp,1.5_dp,2._dp,2._dp]
  model=quantile_transformer();model%n_quantiles=9;model%subsample=0;z=model%fit_transform(ties)
  call require(all(z(4:6,1)==.5_dp),'exact repeated interior knots average both directions')
  model%output_distribution='normal';z=model%transform(ties)
  call require(all(z(4:6,1)==0),'exact repeated median knots map to normal zero')
  print *, 'QuantileTransformer tests passed'
end program
