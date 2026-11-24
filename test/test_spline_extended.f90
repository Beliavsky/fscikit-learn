program test_spline_extended
 use sklearn,only:dp,spline_transformer,csr_matrix,feature_string
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan
 implicit none
 type(spline_transformer)::model
 type(csr_matrix)::s
 type(feature_string)::input(1)
 type(feature_string),allocatable::names(:)
 real(dp)::x(7,1),q(4,1),w(7),saved(7,1)
 real(dp),allocatable::z(:,:),dense(:,:),again(:,:)
 integer,allocatable::before(:),after(:)
 integer::n
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 x(:,1)=[-100._dp,0._dp,1._dp,2._dp,3._dp,4._dp,100._dp];saved=x
 w=[0._dp,1._dp,1._dp,1._dp,1._dp,1._dp,0._dp]
 model%n_knots=3;model%degree=1;input(1)%text='income'
 call model%fit(x,sample_weight=w,feature_names=input)
 call require(all(model%base_knots_(:,1)==[0._dp,2._dp,4._dp]),'weighted uniform ignores zero-weight extrema')
 call require(all(x==saved),'spline fit preserves caller input')
 names=model%get_feature_names_out()
 call require(names(1)%text=='income_sp_0'.and.names(3)%text=='income_sp_2','spline fitted names')
 q(:,1)=[0._dp,1._dp,2._dp,4._dp];s=model%transform_csr(q);dense=s%to_dense();z=model%transform(q)
 call require(maxval(abs(dense-z))==0,'spline dense and sparse maps')
 call require(s%nnz()==8.and.count(s%data==0)==3,'spline CSR retains active-support zero slots')
 s%data=100;again=model%transform(q);call require(all(again==z),'CSR output owns storage')
 model%extrapolation='linear';q(:,1)=[-2._dp,0._dp,4._dp,6._dp];z=model%transform(q)
 call require(all(z(1,:)==[2._dp,-1._dp,0._dp]),'spline live linear extrapolation')
 model%degree=0;model%knots='quantile';model%extrapolation='continue'
 z=model%fit_transform(x,sample_weight=w)
 call require(all(model%base_knots_(:,1)==[0._dp,2._dp,4._dp]),'weighted lower quantiles')
 call require(all(sum(z,dim=2)==1),'degree-zero interval basis')
 call require(.not.allocated(model%feature_names_in_),'unnamed refit clears feature names')
 names=model%get_feature_names_out();call require(names(1)%text=='x0_sp_0','default spline names')
 x(:,1)=[0._dp,0._dp,0._dp,1._dp,2._dp,3._dp,4._dp];model%degree=3;model%n_knots=4
 model%knots='quantile';model%extrapolation='linear';z=model%fit_transform(x)
 call require(all(abs(sum(z,dim=2)-1)<1.e-14_dp),'repeated generated knots retain partition')
 model%extrapolation='periodic';x=1;z=model%fit_transform(x)
 call require(all(ieee_is_nan(z)),'zero-period dense source NaNs')
 model=spline_transformer();model%degree=2;model%n_knots=4;model%include_bias=.false.
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
 s=model%fit_transform_csr(x);z=model%transform(x)
 dense=s%to_dense();call require(maxval(abs(dense-z))<1.e-14_dp,'direct sparse fit transform')
 call random_seed(get=after);call require(all(before==after),'spline preserves caller RNG')
 print *, 'Extended spline tests PASS'
end program
