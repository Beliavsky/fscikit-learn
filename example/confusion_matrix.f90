program confusion_matrix_example
 use sklearn,only:dp,confusion_matrix
 implicit none
 integer::y(6),p(6),i
 integer,allocatable::counts(:,:)
 real(dp),allocatable::normalized(:,:)
 y=[2,0,2,2,0,1];p=[0,0,2,2,0,2]
 counts=confusion_matrix(y,p)
 print *, 'Integer confusion matrix:'
 do i=1,size(counts,1)
  print *, counts(i,:)
 end do
 normalized=confusion_matrix(y,p,normalize='true',sample_weight=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp])
 print *, 'Weighted row-normalized confusion matrix:'
 do i=1,size(normalized,1)
  print *, normalized(i,:)
 end do
end program
