program test_order_errors
 use sklearn,only:dp,median_absolute_error,median_absolute_error_raw,max_error
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),v,h
 real(dp),allocatable::raw(:)
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
 call require(median_absolute_error(y,p)==0,'legacy median')
 call require(median_absolute_error(y,p,sample_weight=[1._dp,4._dp,1._dp])==1,'weighted median')
 call require(median_absolute_error([0._dp,0._dp],[0._dp,2._dp])==1,'ordinary midpoint median')
 call require(median_absolute_error([0._dp,0._dp],[0._dp,2._dp],sample_weight=[1._dp,1._dp])==0, &
  'weighted lower median differs from ordinary midpoint')
 call require(median_absolute_error(y,p,sample_weight=[0._dp,0._dp,0._dp])==1,'all zero weights select maximum')
 call require(median_absolute_error([0._dp],[3._dp],sample_weight=[0._dp])==3,'singleton zero weight')
 call require(median_absolute_error([0._dp,0._dp],[0._dp,2._dp],sample_weight=[-1._dp,1._dp])==2, &
  'signed zero-total CDF search')
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 v=median_absolute_error(a,b,sample_weight=[1._dp,4._dp,1._dp],multioutput=[1._dp,3._dp],raw_values=raw)
 call require(v==.25_dp.and.all(raw==[1._dp,0._dp]),'weighted output median and owned raw values')
 raw=median_absolute_error_raw(a,b,sample_weight=[1._dp,4._dp,1._dp])
 call require(size(raw)==2.and.raw(1)==1.and.raw(2)==0,'owned raw helper')
 call require(max_error(y,p)==1,'legacy maximum')
 call require(max_error(reshape(y,[3,1]),p)==1,'column/vector maximum')
 call require(median_absolute_error(y,reshape(p,[3,1]))==0,'mixed ranks')
 h=huge(1._dp)
 call require(median_absolute_error([0._dp,0._dp,0._dp],[h,h,h])==h,'odd median avoids spurious overflow')
 call require(median_absolute_error([0._dp,0._dp],[h,h])>h,'even midpoint retains source overflow')
 call require(max_error([h],[-h])>h,'maximum quietly propagates overflow')
 call random_seed(get=after);call require(all(before==after),'order metrics preserve RNG')
 print *, 'Weighted multioutput median and dense max error: PASS'
end program
