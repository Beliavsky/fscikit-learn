program test_hamming_loss
 use sklearn,only:dp,hamming_loss,csr_matrix,csr_from_dense
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 integer::y(2,3),p(2,3),n,i,extra_y(5,3),extra_p(5,3)
 integer,allocatable::truth(:),guess(:),rng_before(:),rng_after(:)
 type(csr_matrix)::a,b
 real(dp)::value,inf
 real(dp),target::storage(10),weights(5)
 logical::flags_before(size(ieee_all)),flags_after(size(ieee_all)),modes_before(size(ieee_all)),modes_after(size(ieee_all))
 call random_seed(size=n);allocate(rng_before(n),rng_after(n));call random_seed(get=rng_before)
 call require(hamming_loss([1,2,3,4],[2,2,3,4])==.25_dp,'legacy categorical Hamming loss')
 value=hamming_loss([0,1,0],[0,0,1],sample_weight=[2._dp,3._dp,5._dp])
 call require(value==.8_dp,'weighted categorical mismatches')
 y=reshape([0,1,1,0,0,1],[2,3]);p=reshape([1,1,1,1,0,0],[2,3])
 call require(hamming_loss(y,p)==.5_dp,'multilabel Hamming counts individual columns')
 value=hamming_loss(y,p,sample_weight=[1._dp,3._dp])
 call require(abs(value-7._dp/12)<1e-15_dp,'weighted multilabel loss')
 call require(hamming_loss(y,p==1)==.5_dp,'mixed logical multilabel target')
 a=csr_from_dense(real(y,dp));b=csr_from_dense(real(p,dp))
 call require(hamming_loss(a,b)==.5_dp.and.hamming_loss(a,p)==.5_dp,'dense and CSR multilabel parity')
 a%n_rows=2;a%n_cols=3;a%indptr=[1,3,4];a%indices=[3,3,1];a%data=[1._dp,1._dp,0._dp]
 b%n_rows=2;b%n_cols=3;b%indptr=[1,2,2];b%indices=[3];b%data=[2._dp]
 call require(hamming_loss(a,b)==0,'sparse duplicates coalesce before counting')
 call require(all(a%data==[1._dp,1._dp,0._dp]),'raw CSR input is preserved')
 call require(ieee_is_nan(hamming_loss([integer::],[integer::])),'empty categorical loss is NaN')
 value=hamming_loss(y,p,sample_weight=[0._dp,0._dp])
 call require(ieee_is_nan(value),'multilabel zero weights do not reject')
 n=100001;allocate(truth(n),guess(n));truth=0;guess=0;guess(n)=1
 call require(hamming_loss(truth,guess)==1._dp/n,'direct categorical loss avoids complement cancellation')
 n=100000;a%n_rows=n;a%n_cols=n;a%indptr=[(i,i=1,n+1)];a%indices=[(i,i=1,n)];a%data=spread(1._dp,1,n)
 b%n_rows=n;b%n_cols=n;b%indptr=spread(1,1,n+1);b%indices=[integer::];b%data=[real(dp)::]
 call require(hamming_loss(a,b)==1e-5_dp,'large sparse loss uses int64 denominator without densifying')
 a%n_rows=0;a%n_cols=2;a%indptr=[1];a%indices=[integer::];a%data=[real(dp)::]
 call require(ieee_is_nan(hamming_loss(a,a,sample_weight=[real(dp)::])),'empty weighted multilabel loss')
 inf=ieee_value(0._dp,ieee_positive_inf)
 extra_y=0;extra_p=0;extra_p(1,:2)=1;extra_p(3,1)=1
 weights=[huge(1._dp),0._dp,-huge(1._dp),0._dp,1._dp];storage=0;storage(1:10:2)=weights
 value=hamming_loss(extra_y,extra_p,sample_weight=weights)
 call require(value==inf,'contiguous Hamming dot retains source overflow')
 value=hamming_loss(extra_y,extra_p,sample_weight=storage(1:10:2))
 call require(value==huge(1._dp)/3,'nonunit Hamming dot uses the fused pair path')
 a%n_rows=1;a%n_cols=2;a%indptr=[1,4];a%indices=[1,1,1]
 a%data=spread(nearest(2._dp**63,-1._dp),1,3);b=a
 call ieee_set_flag(ieee_all,.false.)
 call ieee_get_flag(ieee_all,flags_before);call ieee_get_halting_mode(ieee_all,modes_before)
 value=hamming_loss(a,b)
 call ieee_get_flag(ieee_all,flags_after);call ieee_get_halting_mode(ieee_all,modes_after)
 call require(all(flags_before.eqv.flags_after).and.all(modes_before.eqv.modes_after),'coalescing preserves IEEE state')
 call require(value==0,'large duplicate values coalesce consistently')
 call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_inexact,.true.)
 call ieee_get_flag(ieee_all,flags_before);call ieee_get_halting_mode(ieee_all,modes_before)
 value=hamming_loss(y,y,sample_weight=[inf,1._dp])
 call ieee_get_flag(ieee_all,flags_after);call ieee_get_halting_mode(ieee_all,modes_after)
 call require(all(flags_before.eqv.flags_after).and.all(modes_before.eqv.modes_after),'Hamming preserves IEEE state')
 call require(ieee_is_nan(value),'nonfinite weights preserve source NaN')
 call random_seed(get=rng_after);call require(all(rng_before==rng_after),'Hamming preserves caller RNG')
 print *, 'Weighted dense and CSR Hamming loss PASS'
end program
