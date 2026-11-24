program test_kbins_extended
 use sklearn,only:dp,kbins_discretizer,csr_matrix,feature_string
 use sklearn_legacy_rng,only:legacy_random_state
 use sklearn_common,only:require
 implicit none
 type(kbins_discretizer)::model
 type(legacy_random_state)::rng
 type(csr_matrix)::s
 type(feature_string)::input(2)
 type(feature_string),allocatable::names(:)
 real(dp)::x(12,2),saved(12,2),soft(1,6),codes(3,2)
 real(dp),allocatable::z(:,:),dense(:,:),back(:,:)
 integer,allocatable::before(:),after(:)
 integer::i,n
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 x(:,1)=[(real(i-1,dp),i=1,12)];x(:,2)=2*x(:,1)+1;saved=x
 input(1)%text='age';input(2)%text='income'
 model%strategy='uniform';model%encode='onehot';model%n_bins=3;model%subsample=4
 call rng%seed(42);s=model%fit_transform_csr(x,feature_names=input,rng=rng)
 call require(model%bin_edges_(1)%values(1)==0.and.model%bin_edges_(1)%values(4)==10, &
  'source sampled extrema')
 call require(rng%raw_uint32()==613608295,'source permutation RNG consumption')
 call require(s%n_rows==12.and.s%n_cols==6.and.s%nnz()==24.and.all(s%data==1),'direct one-hot CSR map')
 dense=s%to_dense();z=model%transform(x);call require(all(dense==z),'onehot dense representation')
 names=model%get_feature_names_out()
 call require(names(1)%text=='age_0.0'.and.names(6)%text=='income_2.0','fitted one-hot names')
 call require(all(x==saved),'bin fit preserves caller input')
 s%data=100;z=model%transform(x);call require(all(sum(z,dim=2)==2),'CSR result owns storage')
 model%encode='ordinal';z=model%transform(x)
 call require(size(z,2)==2,'live ordinal encoding')
 names=model%get_feature_names_out();call require(size(names)==6,'cached encoder still supplies names')
 model%subsample=0;model%n_bins=2;call model%fit(x)
 call require(model%n_features_out_==2.and..not.allocated(model%feature_names_in_),'ordinal unnamed refit metadata')
 model%encode='onehot-dense';z=model%transform(x)
 call require(size(z,2)==6.and.all(sum(z,dim=2)==2),'ordinal refit retains old one-hot categories')
 soft(1,:)=[.1_dp,.7_dp,.2_dp,-1._dp,-3._dp,-2._dp];back=model%inverse_transform(soft)
 call require(maxval(abs(back(1,:)-[8.25_dp,6.5_dp]))<1.e-12_dp,'soft and signed one-hot inverse argmax')
 model%encode='ordinal';codes(1,:)=[-.7_dp,-1.2_dp];codes(2,:)=[1.9_dp,-2.1_dp];codes(3,:)=[-2.9_dp,.4_dp]
 back=model%inverse_transform(codes)
 call require(all(back(1,:)==[2.75_dp,17.5_dp]).and.all(back(2,:)==[8.25_dp,6.5_dp]), &
  'fractional truncation and Python negative indices')
 model%strategy='quantile';model%n_bins=4;z=model%fit_transform(x*1.e-12_dp)
 call require(all(model%n_bins_==0).and.all(z==0),'source ordinal zero bins after width pruning')
 call random_seed(get=after);call require(all(before==after),'bin operations preserve caller global RNG')
 print *, 'Extended KBins tests PASS'
end program
