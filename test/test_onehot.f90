program test_onehot
 use sklearn,only:dp,one_hot_encoder,onehot_inverse_result,onehot_name_value,feature_string,csr_matrix
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_value,ieee_quiet_nan
 implicit none
 type(one_hot_encoder)::model
 type(onehot_inverse_result)::decoded
 type(csr_matrix)::s
 type(feature_string)::input(2)
 type(feature_string),allocatable::names(:)
 real(dp)::x(6,2),q(2,2),saved(6,2)
 real(dp),allocatable::z(:,:),a(:,:)
 integer,allocatable::before(:),after(:)
 integer::n,callback_calls
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 x(:,1)=[0._dp,0._dp,0._dp,1._dp,2._dp,ieee_value(0._dp,ieee_quiet_nan)]
 x(:,2)=[1._dp,1._dp,2._dp,2._dp,2._dp,3._dp];saved=x
 input(1)%text='color';input(2)%text='level'
 model%min_frequency_count=2;model%handle_unknown='infrequent_if_exist'
 s=model%fit_transform_csr(x,input);z=s%to_dense();a=model%transform(x)
 call require(all(z==a).and.all(sum(z,dim=2)==2),'owned dense/CSR grouped feature maps')
 call require(model%n_features_out_==5.and.model%n_features_in_==2,'grouped feature widths')
 call require(size(model%infrequent_categories_(1)%values)==3,'NaN is included in rare categories')
 decoded=model%inverse_transform(s)
 call require(count(decoded%is_infrequent)==4.and..not.any(decoded%is_none),'infrequent inverse object mask')
 call require(all(decoded%values(1,:)==[0._dp,1._dp]),'frequent numeric inverse values')
 names=model%get_feature_names_out()
 call require(names(2)%text=='color_infrequent_sklearn'.and.names(5)%text=='level_infrequent_sklearn', &
  'infrequent names use source token')
 q(1,:)=[99._dp,1._dp];q(2,:)=[2._dp,99._dp];z=model%transform(q)
 call require(all(sum(z,dim=2)==2),'unknown values map to fitted rare category')
 model%handle_unknown='ignore';z=model%transform(q);decoded=model%inverse_transform(z)
 call require(decoded%is_none(1,1).and.decoded%is_none(2,2),'ignored unknowns are distinct None values')
 model%handle_unknown='error';model%min_frequency_count=0;call model%fit(x)
 z=model%transform(x);decoded=model%inverse_transform(z);a=decoded%as_numeric()
 call require(ieee_is_nan(a(6,1)).and..not.any(decoded%is_none),'known NaN differs from None')
 model%drop='first';model%handle_unknown='ignore';call model%fit(x,input)
 call require(all(model%drop_idx_==1),'drop indices are one-based metadata')
 z=model%transform(x);decoded=model%inverse_transform(z)
 call require(all(decoded%values(1,:)==[0._dp,1._dp]),'all-zero dropped categories invert')
 callback_calls=0;model%feature_name_combiner=>combine
 names=model%get_feature_names_out()
 call require(callback_calls==size(names)+1,'name callback includes source validation call')
 call require(names(1)%text=='color__numeric','custom feature names')
 model%feature_name_combiner=>null();model%drop='none';model%min_frequency_count=2
 call model%fit(x);call require(.not.allocated(model%feature_names_in_),'unnamed refit clears names')
 s=model%transform_csr(x);s%data=10;z=model%transform(x)
 call require(all(sum(z,dim=2)==2),'CSR storage does not alias fitted categories')
 call require(all(ieee_is_nan(x).eqv.ieee_is_nan(saved)),'fit preserves caller missing mask')
 where(ieee_is_nan(saved)) saved=0
 a=x;where(ieee_is_nan(a)) a=0
 call require(all(a==saved),'fit preserves caller values')
 call random_seed(get=after);call require(all(before==after),'one-hot operations preserve caller RNG')
 print *, 'OneHotEncoder numeric/drop/infrequent/CSR tests PASS'
contains
 function combine(feature,category) result(text)
  character(*),intent(in)::feature
  type(onehot_name_value),intent(in)::category
  type(feature_string)::text
  callback_calls=callback_calls+1
  text%text=feature//'__numeric'
  if(category%is_string) text%text=feature//'__text'
 end function
end program
