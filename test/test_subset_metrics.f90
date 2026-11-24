program test_subset_metrics
 use sklearn,only:dp,accuracy_score,zero_one_loss,csr_matrix
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 type(csr_matrix)::a,b
 integer::i,n,y(2,2),p(2,2),zeros16(16)
 integer,allocatable::truth(:),guess(:),before(:),after(:)
 real(dp),allocatable::w(:)
 real(dp),target::storage(32)
 real(dp)::large,pattern(16),value
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 call require(abs(accuracy_score([1,0,2],[1,2,2])-2._dp/3)<1e-15_dp,'legacy integer accuracy')
 call require(accuracy_score([1,0,2],[1,2,2],normalize=.false.)==2,'unnormalized accuracy')
 call require(zero_one_loss([1,0,2],[1,2,2],normalize=.false.)==1,'zero-one count')
 value=accuracy_score([1,0,2],[1,2,2],sample_weight=[1._dp,2._dp,4._dp])
 call require(abs(value-5._dp/7)<1e-15_dp,'weighted normalized accuracy')
 value=accuracy_score([1,0,2],[1,2,2],sample_weight=[1._dp,-2._dp,4._dp])
 call require(abs(value-5._dp/3)<1e-15_dp,'signed weights retain source values')
 value=accuracy_score([1,0,0],[.true.,.false.,.true.])
 call require(abs(value-2._dp/3)<1e-15_dp,'mixed logical and integer vectors')
 value=accuracy_score(reshape([1,0,2],[3,1]),[1._dp,2._dp,2._dp])
 call require(abs(value-2._dp/3)<1e-15_dp,'single-column mixed-kind targets')
 y=reshape([2,3,3,2],[2,2]);p=y;p(2,1)=2
 call require(accuracy_score(y,p)==.5_dp,'nonbinary indicator values are not booleanized')
 a%n_rows=2;a%n_cols=3;a%indptr=[1,3,4];a%indices=[3,3,1];a%data=[1._dp,1._dp,0._dp]
 b%n_rows=2;b%n_cols=3;b%indptr=[1,2,2];b%indices=[3];b%data=[2._dp]
 call require(accuracy_score(a,b)==1,'CSR duplicates coalesce before matching')
 call require(accuracy_score(a,int(b%to_dense()))==1,'CSR and dense mixed indicators')
 call require(all(a%data==[1._dp,1._dp,0._dp]),'subset metrics preserve caller storage')
 call require(ieee_is_nan(accuracy_score([integer::],[integer::])),'empty accuracy is NaN')
 call require(zero_one_loss([integer::],[integer::],normalize=.false.)==0,'empty count is zero')
 value=accuracy_score([0,0],[0,0],sample_weight=[ieee_value(0._dp,ieee_positive_inf),1._dp])
 call require(ieee_is_nan(value),'nonfinite weights follow source IEEE behavior')
 large=huge(1._dp);zeros16=0
 do i=1,16
  pattern(i)=merge(large,-large,mod(i-1,4)<2)
 end do
 storage=0;storage(1:32:2)=pattern
 value=accuracy_score(zeros16,zeros16,normalize=.false.,sample_weight=pattern)
 call require(ieee_is_nan(value),'contiguous masked dot follows source lane overflow')
 value=accuracy_score(zeros16,zeros16,normalize=.false.,sample_weight=storage(1:32:2))
 call require(value==0,'positive-stride masked dot follows source two-accumulator path')
 n=10001;allocate(truth(n),guess(n),w(n));truth=0;guess=0;guess(1:n:5)=1
 do i=1,n
  select case(mod(i-1,4))
  case(0);w(i)=1e16_dp
  case(1);w(i)=1
  case(2);w(i)=-1e16_dp
  case(3);w(i)=3
  end select
 end do
 value=zero_one_loss(truth,guess,normalize=.false.,sample_weight=w)
 call require(value==10000000000002112._dp,'NumPy buffered weight reduction')
 n=100000;a%n_rows=n;a%n_cols=n;a%indptr=[(i,i=1,n+1)];a%indices=[(i,i=1,n)];a%data=spread(1._dp,1,n)
 call require(accuracy_score(a,a)==1,'large CSR target stays compact')
  call require(zero_one_loss(a,a,normalize=.false.)==0,'large sparse zero-one loss')
 a%n_rows=3;a%n_cols=100000000;a%indptr=[1,2,2,3];a%indices=[1,100000000];a%data=[1._dp,1._dp]
 call require(accuracy_score(a,a)==1,'extremely wide canonical CSR avoids column scratch storage')
 call random_seed(get=after);call require(all(before==after),'subset metrics preserve caller RNG')
 print *, 'Accuracy and zero-one subset metrics PASS'
end program
