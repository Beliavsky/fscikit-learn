program test_simple_imputer
 use sklearn,only:dp,simple_imputer,pipeline,frozen_estimator
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use sklearn_feature_extraction,only:feature_string
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_value,ieee_quiet_nan
 implicit none
 type(simple_imputer),target::model
 type(csr_matrix)::x,z,big
 type(pipeline)::graph
 type(frozen_estimator)::view
 type(feature_string),allocatable::names(:)
 real(dp)::a(3,3),nan
 real(dp),allocatable::mapped(:,:),back(:,:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 a(:,1)=[1._dp,nan,3._dp];a(:,2)=nan;a(:,3)=[4._dp,5._dp,6._dp]
 model%add_indicator=.true.;mapped=model%fit_transform(a)
 call require(size(mapped,2)==4.and.ieee_is_nan(model%statistics_(2)),'empty feature dropping and appended indicators')
 call require(all(model%indicator_%features_==[1,2]),'learned indicator columns are one-based')
 call require(mapped(2,1)==2.and.all(mapped(:,4)==1),'mean imputation and empty-column indicator')
 back=model%inverse_transform(mapped)
 call require(ieee_is_nan(back(2,1)).and.all(ieee_is_nan(back(:,2))).and.all(back(:,3)==a(:,3)), &
  'inverse reconstruction follows source indicator mapping')
 names=model%get_feature_names_out()
 call require(size(names)==4.and.names(2)%text=='x2'.and.names(4)%text=='missingindicator_x1','dropped/indicator names')
 model%keep_empty_features=.true.;call model%fit(a)
 call require(model%statistics_(2)==0,'kept empty features receive zero statistics')
 model=simple_imputer();model%missing_values=-999;model%strategy='median';a(2,1)=-999;a(:,2)=0
 mapped=model%fit_transform(a);call require(mapped(2,1)==2,'finite missing marker')
 x=csr_from_arrays(4,2,[1,3,4,5,6],[1,1,2,1,2],[1._dp,nan,4._dp,3._dp,nan])
 model=simple_imputer();model%add_indicator=.true.;z=model%fit_transform(x)
 call require(maxval(abs(model%statistics_-[4._dp/3,4._dp/3]))<1.e-14_dp,'sparse raw means include implicit zeros')
 call require(z%n_cols==4.and.z%nnz()==7,'sparse replacement and raw indicators')
 call require(z%indices(1)==1.and.z%indices(2)==1.and.z%indices(3)==3,'raw duplicate slots retained')
 model=simple_imputer();model%strategy='callable';model%statistic=>count_values;model%keep_empty_features=.true.
 a=nan;call model%fit(a);call require(all(model%statistics_==2),'dense callable receives empty columns')
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1));big%indptr=2;big%indptr(1)=1
 big%indices=100000;big%data=nan;call model%fit(big)
 call require(all(model%statistics_==0),'sparse callable keep-empty shortcut ignores implicit zeros')
 model=simple_imputer();model%add_indicator=.true.;a(:,1)=[1._dp,nan,3._dp];a(:,2)=nan;a(:,3)=1
 allocate(graph%steps(1));graph%steps(1)%name='impute';allocate(graph%steps(1)%model,source=model)
 mapped=graph%fit_transform(a);names=graph%get_feature_names_out()
 call require(size(mapped,2)==4.and.size(names)==4.and.graph%can_inverse(),'pipeline imputer names/inverse capability')
 back=graph%inverse_transform(mapped);call require(all(ieee_is_nan(back(:,2))),'pipeline imputer inverse')
 call model%fit(a);view%estimator=>model
 call require(view%is_fitted().and.view%can_inverse(),'frozen imputer inverse capability')
 back=view%inverse_transform(mapped);call require(all(ieee_is_nan(back(:,2))),'frozen imputer inverse delegation')
 print *, 'SimpleImputer dense/CSR, empty features, markers, indicators, callbacks and graph dispatch: PASS'
contains
 function count_values(v) result(value)
  real(dp),intent(in)::v(:)
  real(dp)::value
  value=real(size(v),dp)+2
 end function
end program
