program test_likelihood_ratios
 use sklearn,only:dp,class_likelihood_ratios
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::ratios(2),nan,inf
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 ratios=class_likelihood_ratios([0,1,0,1,0],[1,1,0,0,0])
 call require(all(abs(ratios-[1.5_dp,.75_dp])<1e-15_dp),'likelihood ratio example')
 ratios=class_likelihood_ratios([0,1,0,1,0],[1,1,0,0,0],labels=[1,0])
 call require(all(abs(ratios-[4._dp/3,2._dp/3])<1e-15_dp),'reordered negative and positive classes')
 ratios=class_likelihood_ratios([0,1],[0,1])
 call require(ieee_is_nan(ratios(1)).and.ratios(2)==0,'perfect classifier has undefined positive ratio')
 ratios=class_likelihood_ratios([1,1],[1,1])
 call require(all(ieee_is_nan(ratios)),'single observed class remains undefined')
 ratios=class_likelihood_ratios([integer::],[integer::],labels=[0,1],sample_weight=[1._dp])
 call require(all(ieee_is_nan(ratios)),'empty explicit-label shortcut precedes weight validation')
 ratios=class_likelihood_ratios(reshape([0,1,0,1,0],[5,1]),[.true.,.true.,.false.,.false.,.false.])
 call require(all(abs(ratios-[1.5_dp,.75_dp])<1e-15_dp),'mixed column and logical targets')
 ratios=class_likelihood_ratios([0,1,0,1,0],[1,1,0,0,0],sample_weight=[2._dp,3._dp,4._dp,5._dp,6._dp])
 call require(all(abs(ratios-[2.25_dp,.75_dp])<1e-15_dp),'weighted likelihood ratios')
 ratios=class_likelihood_ratios([0,1,0,1],[0,1,1,0],labels=[0,0])
 call require(all(ieee_is_nan(ratios)),'duplicate selected labels retain source mapping')
 nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 ratios=class_likelihood_ratios([0,0,1,1],[0,1,0,1],sample_weight=[nan,1._dp,1._dp,1._dp])
 call require(ratios(1)==.5_dp.and.ieee_is_nan(ratios(2)),'NaN confusion cells are replaced with zero')
 ratios=class_likelihood_ratios([0,0,1,1],[0,1,0,1],sample_weight=[inf,inf,inf,inf])
 call require(all(ieee_is_nan(ratios)),'quiet overflow in ratio products')
 call random_seed(get=after);call require(all(before==after),'likelihood ratios preserve caller RNG')
 print *, 'Class likelihood ratios PASS'
end program
