program test_categorical_scores
 use sklearn,only:dp,balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 integer::n,i,y(6),p(6)
 integer,allocatable::rng_before(:),rng_after(:)
 real(dp)::value,w(6),inf
 logical::before(size(ieee_all)),after(size(ieee_all)),modes_before(size(ieee_all)),modes_after(size(ieee_all))
 call random_seed(size=n);allocate(rng_before(n),rng_after(n));call random_seed(get=rng_before)
 y=[0,1,0,0,1,0];p=[0,1,0,0,0,1]
 call require(balanced_accuracy_score(y,p)==.625_dp,'legacy balanced accuracy')
 call require(balanced_accuracy_score(y,p,adjusted=.true.)==.25_dp,'chance-adjusted balanced accuracy')
 call require(matthews_corrcoef(y,p)==.25_dp,'legacy Matthews correlation')
 call require(abs(cohen_kappa_score(y,p)-.25_dp)<1e-15_dp,'legacy Cohen kappa')
 y=[0,1,2,2,1,0];p=[0,2,2,1,1,0];w=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
 value=balanced_accuracy_score(y,p,sample_weight=w)
 call require(abs(value-5._dp/7)<1e-15_dp,'weighted multiclass balanced accuracy')
 value=matthews_corrcoef(y,p,sample_weight=w)
 call require(abs(value-.5793654595023211_dp)<1e-14_dp,'weighted multiclass Matthews correlation')
 value=cohen_kappa_score(y,p,weights='linear',sample_weight=w)
 call require(abs(value-2._dp/3)<1e-15_dp,'linear weighted kappa')
 value=cohen_kappa_score(y,p,weights='quadratic',sample_weight=w)
 call require(abs(value-10._dp/13)<1e-15_dp,'quadratic weighted kappa')
 value=cohen_kappa_score(y,p,labels=[2,0,1],weights='quadratic',sample_weight=w)
 call require(abs(value-1._dp/7)<1e-15_dp,'explicit kappa class order affects ordinal distance')
 value=balanced_accuracy_score(reshape([0,1,0,0,1,0],[6,1]),[.false.,.true.,.false.,.false.,.false.,.true.])
 call require(value==.625_dp,'mixed logical and integer column targets')
 call require(ieee_is_nan(balanced_accuracy_score([integer::],[integer::])),'empty balanced accuracy is NaN')
 call require(matthews_corrcoef([integer::],[integer::])==0,'empty Matthews correlation is zero')
 call require(ieee_is_nan(cohen_kappa_score([integer::],[integer::])),'empty kappa is NaN')
 call require(ieee_is_nan(cohen_kappa_score([1,1],[1,1])),'constant-class kappa retains source NaN')
 value=cohen_kappa_score([integer::],[integer::],labels=[0,1],sample_weight=[1._dp])
 call require(ieee_is_nan(value),'kappa empty explicit-label shortcut precedes weight validation')
 value=balanced_accuracy_score([0,0,1,1],[0,1,0,1],sample_weight=[0._dp,0._dp,1._dp,1._dp])
 call require(value==.5_dp,'zero-support class recalls are removed')
 call require(matthews_corrcoef([0,1],[0,1],sample_weight=[0._dp,0._dp])==0,'zero-weight MCC')
 inf=ieee_value(0._dp,ieee_positive_inf)
 do i=1,2
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_inexact,i==1)
  call ieee_get_flag(ieee_all,before);call ieee_get_halting_mode(ieee_all,modes_before)
  value=matthews_corrcoef([0,0,1,1],[0,1,0,1],sample_weight=[inf,inf,inf,inf])
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,modes_after)
  call require(all(before.eqv.after).and.all(modes_before.eqv.modes_after),'caller IEEE state is preserved')
  call require(ieee_is_nan(value),'exceptional MCC returns quiet NaN')
 end do
 call random_seed(get=rng_after);call require(all(rng_before==rng_after),'categorical scores preserve caller RNG')
 print *, 'Weighted categorical scores PASS'
end program
