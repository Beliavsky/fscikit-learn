program test_neighbor_selection
 use sklearn_common,only:dp,require
 use sklearn_neighbor_selection,only:argkmin_order,pairwise_knn_order
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_positive_inf
 implicit none
 real(dp)::values(9),selected(3)
 integer::indices(3)
 ! Expected orders from sklearn 1.6.1's actual serial Manhattan ArgKmin
 ! and NumPy 2.2.6 argpartition/argsort, not training-index tie breaking.
 values=[0._dp,0._dp,0._dp,0._dp,1._dp,1._dp,1._dp,2._dp,3._dp]
 call require(all(argkmin_order(values,3,9)==[2,3,1]),'parallel-on-Y heap merge ties')
 call require(all(argkmin_order(values,3,10)==[3,1,2]),'parallel-on-X heap ties')
 call require(all(pairwise_knn_order(values,3)==[1,2,3]),'NumPy fallback ties')
 values=ieee_value(0._dp,ieee_positive_inf)
 indices=argkmin_order(values,3,9,selected)
 call require(all(indices==1).and.all(selected==huge(1._dp)),'finite ArgKmin overflow sentinel')
 print *,'neighbor selection tests passed'
end program
