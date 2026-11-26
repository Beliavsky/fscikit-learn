! Scalar SciPy OpenBLAS 0.3.28 SkylakeX TN accumulation for neighbor reductions.
! Copyright (c) 2021, The OpenBLAS Project; retained OpenBLAS BSD notices.
! K blocking: Copyright 2009, 2010 The University of Texas at Austin.
! See LICENSES/OpenBLAS-small-kernel-BSD, LICENSES/OpenBLAS-level3-BSD and notices.
! Native shape/one-based adapters are MIT; no BLAS runtime is linked.
module sklearn_neighbor_gram
 use sklearn_common,only:dp
 use sklearn_kernel_dispatch,only:kernel_blas_inner_product
 use,intrinsic::ieee_arithmetic,only:ieee_fma
 implicit none
 private
 public::neighbor_row_norm,neighbor_middle_term
contains
 real(dp)function neighbor_row_norm(row)result(value)
  real(dp),intent(in)::row(:)
  value=kernel_blas_inner_product(row,row)
 end function
 real(dp)function neighbor_middle_term(u,v,rows,columns,i,j)result(value)
  real(dp),intent(in)::u(:),v(:)
  integer,intent(in)::rows,columns,i,j
  real(dp)::lane(8),part
  integer::k,l,first,count
  value=0
  if(size(u)>=32.and.real(rows,dp)*columns<=1200.and.real(rows,dp)*columns*size(u)<=1.e6_dp)then
   lane=0
   do k=1,size(u)
    l=mod(k-1,8)+1;lane(l)=ieee_fma(u(k),v(k),lane(l))
   end do
   if(i>rows-mod(rows,4).and.j>columns-mod(columns,4))then
    ! Bottom/right tails use _mm512_reduce_add_pd rather than REDUCE_4.
    value=((lane(1)+lane(5))+(lane(3)+lane(7)))+((lane(2)+lane(6))+(lane(4)+lane(8)))
   else
    value=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
   end if
  else
   first=1
   do while(first<=size(u))
    count=size(u)-first+1
    if(count>=768)then
     count=384
    else if(count>384)then
     count=((count/2+15)/16)*16
    end if
    part=0
    do k=first,first+count-1
     part=ieee_fma(u(k),v(k),part)
    end do
    value=ieee_fma(-2._dp,part,value);first=first+count
   end do
   return
  end if
  value=-2*value
 end function
end module
