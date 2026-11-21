program test_target_encoding
  use sklearn, only: dp,target_encoder,target_category
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(target_encoder) :: enc,copy
  real(dp) :: x(6,1),y(6),nan
  real(dp), allocatable :: z(:,:),full(:,:),query(:,:)
  integer :: i
  nan=ieee_value(0._dp,ieee_quiet_nan)
  x(:,1)=[1._dp,1._dp,2._dp,2._dp,3._dp,3._dp]; y=[0._dp,2._dp,4._dp,6._dp,8._dp,10._dp]
  enc%target_type='continuous'; enc%auto_smooth=.false.; enc%smooth=2; enc%cv=3; enc%shuffle=.false.
  z=enc%fit_transform(x,y); full=enc%transform(x)
  call require(maxval(abs(full(:,1)-[3._dp,3._dp,5._dp,5._dp,7._dp,7._dp]))<1.e-14_dp, &
    'TargetEncoder shrinkage')
  call require(maxval(abs(z(:,1)-[7._dp,7._dp,5._dp,5._dp,3._dp,3._dp]))<1.e-14_dp, &
    'TargetEncoder held-out category uses fold mean')
  call require(all(enc%cv_splits_(1)%test==[1,2]),'TargetEncoder sorted validation indices')
  query=enc%transform(reshape([99._dp,nan],[2,1]))
  call require(all(query==5),'TargetEncoder unknown and unseen NaN use global mean')
  copy=enc; enc%smooth=0; enc%target_type='binary'
  query=enc%transform(x); call require(all(query==full),'TargetEncoder learned state survives parameter mutation')
  enc=target_encoder(); enc%cv=3; x(1,1)=nan
  z=enc%fit_transform(x,[8,-3,8,-3,8,-3])
  call require(enc%target_type_=='binary'.and.all(enc%classes_==[-3._dp,8._dp]),'TargetEncoder class encoding')
  call require(size(enc%categories_(1)%values)==4,'TargetEncoder NaN is a category')
  query=enc%transform(reshape([nan],[1,1])); call require(query(1,1)==1,'TargetEncoder fitted NaN encoding')
  x(:,1)=[(real(i,dp),i=1,6)]; enc%target_type='multiclass'; enc%shuffle=.false.; enc%cv=2
  z=enc%fit_transform(x,[8,-3,4,8,-3,4])
  call require(all(shape(z)==[6,3]),'TargetEncoder feature-major multiclass width')
  call require(maxval(abs(sum(z,dim=2)-1))<1.e-14_dp,'TargetEncoder multiclass fold means sum to one')
  full=enc%transform(x)
  call require(full(1,3)==1.and.full(2,1)==1.and.full(3,2)==1,'TargetEncoder multiclass sorted class columns')
  allocate(enc%categories(1)); enc%categories(1)%values=[0._dp,1._dp,10._dp,nan]
  enc%target_type='continuous'; enc%auto_smooth=.false.; enc%smooth=0
  call enc%fit(x,y); query=enc%transform(reshape([0._dp,1._dp,2._dp,nan],[4,1]))
  call require(all(query(:,1)==[6._dp,0._dp,5._dp,5._dp]),'TargetEncoder specified categories and zero smoothing')
  call require(.not.allocated(enc%cv_splits_),'TargetEncoder fit clears cross-fitting diagnostics')
  enc%cv=3
  z=enc%fit_transform(x,y)
  call require(all(z(:,1)==[7._dp,3._dp,3._dp,3._dp,3._dp,3._dp]), &
    'TargetEncoder source omitted-category cross-fitting mask behavior')
  enc=copy; enc%auto_smooth=.true.; call enc%fit(x,spread(17._dp,1,6)); full=enc%transform(x)
  call require(all(full==17),'TargetEncoder constant-target automatic smoothing')
  print *, 'Target encoding: PASS'
end program
