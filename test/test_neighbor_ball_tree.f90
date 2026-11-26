program test_neighbor_ball_tree
 use sklearn_common,only:dp,require
 use sklearn_neighbor_ball_tree,only:neighbor_ball_tree
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(neighbor_ball_tree)::tree
 real(dp)::x(65,2),zero(4,2)
 real(dp),allocatable::centers(:,:),radii(:),distance(:)
 integer,allocatable::indices(:),spans(:,:),ids(:)
 integer::i
 logical::flags(size(ieee_all)),modes(size(ieee_all)),after_flags(size(ieee_all)),after_modes(size(ieee_all))
 ! Actual sklearn 1.6.1 BallTree partition, leaf_size=30, integer coordinates.
 integer,parameter::expected(65)=[1,34,31,4,32,63,7,62,41,10,60,59,13,56,53,16,50,65,19,47,44,22,23, &
  38,25,26,35,28,29,57,20,54,17,51,14,48,11,45,8,42,5,39,2,3,30,27,36,37,18,9,46,52,15,21,55,43,33, &
  58,12,40,61,24,6,64,49]
 do i=1,65
  x(i,:)=[real(mod(17*i,67),dp),real(mod(23*i,71),dp)]
 end do
 call tree%fit(x);call tree%get_arrays(indices,spans,centers,radii)
 call require(all(indices==expected),'Source BallTree partial partition order')
 call require(all(spans(1,:)==[1,65,0]).and.all(spans(2,:)==[1,32,1]).and. &
  all(spans(3,:)==[33,65,1]),'Ball tree spans')
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 call tree%query([0._dp,0._dp],3,distance,ids)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'tree query preserves IEEE state')
 call require(all(ids==[28,16,56]),'Source depth-first nearest query')
 call require(maxval(abs(distance-sqrt([74._dp,185._dp,296._dp])))<1.e-12_dp,'Ball tree query distances')
 call tree%query_radius([0._dp,0._dp],ieee_value(0._dp,ieee_positive_inf),distance,ids)
 call require(all(ids==expected),'Source radius storage order')
 call tree%query_radius(x(1,:),0._dp,distance,ids)
 call require(size(ids)==1.and.ids(1)==1.and.distance(1)==0,'inclusive zero radius')
 zero=0;call tree%fit(zero,leaf_size=1);call tree%query([0._dp,0._dp],3,distance,ids)
 call require(all(ids==[3,1,2]).and.all(distance==0),'tree heap ties without ArgKmin merge')
 call tree%query_radius([0._dp,0._dp],-1._dp,distance,ids)
 call require(size(ids)==0,'negative BallTree radius returns no points')
 call tree%query_radius([0._dp,0._dp],ieee_value(0._dp,ieee_quiet_nan),distance,ids)
 call require(size(ids)==0,'NaN BallTree radius follows Source comparisons')
 print *,'Ball partition, depth-first queries, radius storage order and IEEE state passed'
end program
