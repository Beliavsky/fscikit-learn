program test_confusion_matrix
 use sklearn,only:dp,confusion_matrix
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 integer::y(6),p(6),n
 integer,allocatable::cm(:,:),expected(:,:),before_rng(:),after_rng(:)
 real(dp),allocatable::values(:,:)
 real(dp)::w(6),inf,nan
 logical::flags_before(size(ieee_all)),flags_after(size(ieee_all)),modes_before(size(ieee_all)),modes_after(size(ieee_all))
 call random_seed(size=n);allocate(before_rng(n),after_rng(n));call random_seed(get=before_rng)
 y=[2,0,2,2,0,1];p=[0,0,2,2,0,2]
 cm=confusion_matrix(y,p);expected=reshape([2,0,1,0,0,0,0,1,2],[3,3])
 call require(all(cm==expected),'legacy integer counts and row orientation')
 call require(kind(confusion_matrix(y,p))==kind(0),'legacy default integer result kind')
 values=confusion_matrix(y,p,normalize='true')
 call require(maxval(abs(values-reshape([1._dp,0._dp,1._dp/3,0._dp,0._dp,0._dp,0._dp,1._dp,2._dp/3],[3,3]))) &
  <1e-15_dp,'true-row normalization')
 values=confusion_matrix(y,p,normalize='pred')
 call require(abs(values(1,1)-2._dp/3)<1e-15_dp.and.all(values(:,2)==0),'predicted-column normalization')
 values=confusion_matrix(y,p,normalize='all')
 call require(maxval(abs(values-real(expected,dp)/6))<1e-15_dp,'whole-matrix normalization')
 w=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
 values=confusion_matrix(y,p,sample_weight=w)
 call require(all(values==real(reshape([7,0,1,0,0,0,0,6,7],[3,3]),dp)),'weighted raw counts')
 values=confusion_matrix(y,p,labels=[2,0],sample_weight=w)
 call require(all(values==real(reshape([7,0,1,7],[2,2]),dp)),'selected class order and excluded samples')
 cm=confusion_matrix(y,p,labels=[0,0,2]);expected=0;expected(2,2)=2;expected(3,2)=1;expected(3,3)=2
 call require(all(cm==expected),'last duplicate label index matches source')
 cm=confusion_matrix([integer::],[integer::]);call require(all(shape(cm)==[0,0]),'empty automatic matrix')
 values=confusion_matrix([integer::],[integer::],labels=[2,0],sample_weight=[1._dp],normalize='pred')
 call require(all(shape(values)==[2,2]).and.all(values==0),'empty explicit-label shortcut')
 cm=confusion_matrix(reshape([0,1,0],[3,1]),[.false.,.true.,.true.],labels=[.false.,.true.])
 call require(all(cm==reshape([1,0,1,1],[2,2])),'mixed targets and logical class order')
 inf=ieee_value(0._dp,ieee_positive_inf);nan=ieee_value(0._dp,ieee_quiet_nan)
 values=confusion_matrix([0,0,1,1],[0,1,0,1],sample_weight=[inf,1._dp,1._dp,1._dp],normalize='true')
 call require(all(values==reshape([0._dp,.5_dp,0._dp,.5_dp],[2,2])),'normalize infinities before sanitizing')
 values=confusion_matrix([0,0,1,1],[0,1,0,1],sample_weight=[nan,1._dp,1._dp,1._dp],normalize='true')
 call require(all(values==reshape([0._dp,.5_dp,0._dp,.5_dp],[2,2])),'normalize NaN cells before sanitizing')
 call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_inexact,.true.)
 call ieee_get_flag(ieee_all,flags_before);call ieee_get_halting_mode(ieee_all,modes_before)
 values=confusion_matrix([0,0,1,1],[0,1,0,1],sample_weight=[inf,1._dp,1._dp,1._dp],normalize='all')
 call ieee_get_flag(ieee_all,flags_after);call ieee_get_halting_mode(ieee_all,modes_after)
 call require(all(flags_before.eqv.flags_after).and.all(modes_before.eqv.modes_after),'confusion matrix IEEE state')
 call require(all(values==0),'all normalization with infinite total')
 call random_seed(get=after_rng);call require(all(before_rng==after_rng),'confusion matrix preserves caller RNG')
 print *, 'Weighted confusion matrices PASS'
end program
