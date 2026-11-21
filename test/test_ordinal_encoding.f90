program test_ordinal_encoding
  use sklearn_ordinal_encoding, only: ordinal_encoder,ordinal_category
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(ordinal_encoder) :: enc,copy
  real(dp) :: x(6,2),nan
  real(dp), allocatable :: z(:,:),restored(:,:),query(:,:)
  logical, allocatable :: unknown(:,:),infrequent(:,:)
  nan=ieee_value(0._dp,ieee_quiet_nan)
  x(:,1)=[2._dp,1._dp,2._dp,3._dp,1._dp,nan]; x(:,2)=[4._dp,4._dp,5._dp,5._dp,6._dp,6._dp]
  z=enc%fit_transform(x)
  call require(all(z(:5,1)==[1._dp,0._dp,1._dp,2._dp,0._dp]).and.ieee_is_nan(z(6,1)), &
    'OrdinalEncoder sorted categories and missing values')
  restored=enc%inverse_transform(z)
  call require(all(restored(:5,:)==x(:5,:)).and.ieee_is_nan(restored(6,1)),'OrdinalEncoder numeric round trip')
  enc%handle_unknown='use_encoded_value'; enc%unknown_value=-1; enc%encoded_missing_value=-2
  query=reshape([99._dp,nan,5._dp,99._dp],[2,2]); z=enc%transform(query)
  call require(all(z(:,1)==[-1._dp,-2._dp]).and.all(z(:,2)==[1._dp,-1._dp]), &
    'OrdinalEncoder live missing and unknown controls')
  restored=enc%inverse_transform(z,unknown,infrequent)
  call require(unknown(1,1).and.unknown(2,2).and..not.any(infrequent),'OrdinalEncoder unknown inverse tags')
  call require(ieee_is_nan(restored(2,1)).and..not.unknown(2,1),'OrdinalEncoder missing differs from unknown')
  copy=enc
  enc=ordinal_encoder(); enc%max_categories=2
  z=enc%fit_transform(x(:5,2:2))
  call require(all(z(:,1)==[1._dp,1._dp,0._dp,0._dp,1._dp]),'OrdinalEncoder frequency grouping')
  call require(all(enc%infrequent_categories_(1)%values==[4._dp,6._dp]), &
    'OrdinalEncoder count ties keep later numeric categories')
  ! Counts are [2,2,1]: category 5 is retained, categories 4 and 6 group.
  call require(all(enc%categories_(1)%values==[4._dp,5._dp,6._dp]),'OrdinalEncoder preserves original categories')
  restored=enc%inverse_transform(z,unknown,infrequent)
  call require(all(infrequent(:,1).eqv.[.true.,.true.,.false.,.false.,.true.]),'OrdinalEncoder grouped inverse tags')
  enc%max_categories=1 ! Group mappings are fixed at fit time.
  query=enc%transform(x(:5,2:2)); call require(all(query==z),'OrdinalEncoder fitted grouping snapshot')
  enc=copy; allocate(enc%categories(2))
  enc%categories(1)%values=[0._dp,1._dp,nan]; enc%categories(2)%values=[4._dp,6._dp,8._dp]
  enc%min_frequency_count=2
  z=enc%fit_transform(x)
  call require(all(z([1,3,4],1)==-1),'OrdinalEncoder omitted explicit training categories stay unknown')
  call require(size(enc%infrequent_categories_(2)%values)==1,'OrdinalEncoder unused explicit categories group')
  enc=ordinal_encoder(); enc%handle_unknown='use_encoded_value'; enc%unknown_value=0; enc%max_categories=1
  z=enc%fit_transform(x(:5,2:2)); restored=enc%inverse_transform(z,unknown,infrequent)
  call require(all(infrequent).and..not.any(unknown),'OrdinalEncoder source group wins sentinel collision')
  enc%max_categories=0; enc%handle_unknown='error'; deallocate(enc%unknown_value)
  z=enc%fit_transform(x)
  call require(enc%n_features_in_==2.and.all([size(enc%infrequent_categories_(1)%values), &
    size(enc%infrequent_categories_(2)%values)]==0),'OrdinalEncoder refit clears grouped state and width')
  print *, 'Ordinal encoding: PASS'
end program
