program test_spline
  use sklearn, only: dp,spline_transformer
  use sklearn_common, only: require
  implicit none
  type(spline_transformer) :: model
  real(dp) :: x(3,1),query(3,1),custom(3,1),wide(3,2)
  real(dp), allocatable :: z(:,:),full(:,:),again(:,:)
  x(:,1)=[0._dp,1._dp,2._dp]; model%degree=1; model%n_knots=3
  query(:,1)=[.5_dp,1.5_dp,2._dp]; call model%fit(x); z=model%transform(query)
  call require(maxval(abs(z(1,:)-[.5_dp,.5_dp,0._dp]))<1.e-14_dp,'analytic linear B-spline basis')
  call require(maxval(abs(z(3,:)-[0._dp,0._dp,1._dp]))<1.e-14_dp,'right endpoint uses last base interval')
  query(:,1)=[-1._dp,1._dp,3._dp]; z=model%transform(query)
  call require(all(z(1,:)==[1._dp,0._dp,0._dp]).and.all(z(3,:)==[0._dp,0._dp,1._dp]), &
    'constant extrapolation clamps basis values')
  model%extrapolation='linear'; call model%fit(x); z=model%transform(query)
  call require(maxval(abs(z(1,:)-[2._dp,-1._dp,0._dp]))<1.e-14_dp,'linear extrapolation uses endpoint derivative')
  model%extrapolation='continue'; call model%fit(x); again=model%transform(query)
  call require(maxval(abs(z-again))<1.e-14_dp,'degree-one polynomial continuation is linear')
  model%extrapolation='periodic'; call model%fit(x); query(:,1)=[.5_dp,2.5_dp,-1.5_dp]
  z=model%transform(query)
  call require(maxval(abs(z-spread(z(1,:),1,3)))<1.e-14_dp,'periodic wrapping on either side of knot range')
  model%include_bias=.false.; model%n_knots=2; call model%fit(x); z=model%transform(query)
  call require(size(z,2)==0,'dropping sole periodic basis yields zero columns')
  model=spline_transformer(); model%n_knots=3; wide(:,1)=x(:,1); wide(:,2)=2*x(:,1)+1
  full=model%fit_transform(wide)
  call require(all(full>=0).and.maxval(abs(sum(full(:,:5),dim=2)-1))<1.e-14_dp,'cubic basis partition of unity')
  call require(maxval(abs(sum(full(:,6:),dim=2)-1))<1.e-14_dp,'independent per-feature basis partition')
  model%include_bias=.false.; call model%fit(wide); z=model%transform(wide)
  call require(maxval(abs(z(:,:4)-full(:,:4)))<1.e-14_dp.and.maxval(abs(z(:,5:)-full(:,6:9)))<1.e-14_dp, &
    'include_bias removes final basis separately for every input feature')
  model%degree=100; model%extrapolation='bad'; again=model%transform(wide)
  call require(all(z==again),'transform retains fitted degree and extrapolation')
  model=spline_transformer(); model%degree=2; custom(:,1)=[0._dp,.5_dp,2._dp]
  call model%fit(x,knots=custom); z=model%transform(x)
  call require(all(model%base_knots_==custom).and.maxval(abs(sum(z,dim=2)-1))<1.e-14_dp, &
    'explicit nonuniform knots retain a complete basis')
  model=spline_transformer(); model%extrapolation='error'
  x(:,1)=[-1.674682944704357_dp,.2_dp,1.767929913579883_dp]
  z=model%fit_transform(x)
  call require(model%base_knots_(5,1)==maxval(x),'uniform placement preserves exact right endpoint')
  call require(maxval(abs(sum(z,dim=2)-1))<1.e-14_dp,'training extrema remain inside error-mode interval')
  print *, 'SplineTransformer tests passed'
end program
