program test_stateless_composition
  use sklearn_common, only: dp,require
  use sklearn_stateless_preprocessing, only: normalizer,binarizer
  use sklearn_pipeline, only: pipeline,pipeline_step
  use sklearn_feature_composition, only: feature_union,column_transformer,transformer_branch
  use sklearn_feature_extraction, only: feature_string
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use sklearn_base_dispatch, only: base_weighted
  implicit none
  type(pipeline) :: sequential
  type(feature_union) :: together
  type(column_transformer) :: columns
  type(normalizer) :: unit
  type(binarizer) :: binary
  type(multinomial_nb) :: mn
  type(bernoulli_nb) :: bn
  type(complement_nb) :: cn
  type(feature_string) :: names(2)
  type(feature_string), allocatable :: result_names(:)
  real(dp) :: x(3,2)
  real(dp), allocatable :: z(:,:),reference(:,:),a(:,:),b(:,:)
  x(1,:)=[3._dp,4._dp];x(2,:)=[-2._dp,1._dp];x(3,:)=0
  names(1)%text='east';names(2)%text='north'
  unit%norm='l1';binary%threshold=.5_dp
  allocate(sequential%steps(2));sequential%steps(1)%name='unit';sequential%steps(2)%name='binary'
  allocate(sequential%steps(1)%model,source=unit);allocate(sequential%steps(2)%model,source=binary)
  a=unit%transform(x);reference=binary%transform(a)
  z=sequential%fit_transform(x,feature_names=names)
  call require(all(z==reference),'pipeline dispatches typed stateless transforms')
  result_names=sequential%get_feature_names_out()
  call require(result_names(1)%text=='east'.and..not.sequential%can_inverse(),'pipeline one-to-one names and no inverse')
  allocate(together%transformer_list(2))
  together%transformer_list(1)%name='unit';together%transformer_list(2)%name='binary'
  allocate(together%transformer_list(1)%model,source=unit);allocate(together%transformer_list(2)%model,source=binary)
  b=binary%transform(x);z=together%fit_transform(x,names)
  call require(all(z(:,:2)==a).and.all(z(:,3:)==b),'feature union dispatch')
  result_names=together%get_feature_names_out()
  call require(result_names(1)%text=='unit__east'.and.result_names(4)%text=='binary__north','feature union names')
  allocate(columns%transformers(2))
  columns%transformers(1)%name='unit';columns%transformers(1)%columns=[2]
  columns%transformers(2)%name='binary';columns%transformers(2)%columns=[1]
  allocate(columns%transformers(1)%model,source=unit);allocate(columns%transformers(2)%model,source=binary)
  z=columns%fit_transform(x,names)
  a=unit%transform(x(:,2:2));b=binary%transform(x(:,1:1))
  call require(all(z(:,1:1)==a).and.all(z(:,2:2)==b),'column transformer dispatch')
  call require(unit%n_features_in_==0.and.binary%n_features_in_==0,'composition prototypes remain unchanged')
  call require(base_weighted(mn).and.base_weighted(bn).and.base_weighted(cn),'discrete NB weighted capability')
  print *, 'Stateless preprocessing composition: PASS'
end program
