!*******************************************************************************
!> license: BSD
!
!  BLAS support routines for LBFGSB.
!  These have been refactored into modern Fortran.

    module lbfgsb_blas_module

#ifndef HAS_BLAS

    use lbfgsb_kinds_module, only: wp => lbfgsb_wp

    use,intrinsic::iso_c_binding,only:c_double
    implicit none

    private

    real(wp),parameter :: zero   = 0.0_wp
    real(wp),parameter :: one    = 1.0_wp
    real(wp),parameter :: two    = 2.0_wp
    real(wp),parameter :: four   = 4.0_wp
    real(wp),parameter :: ten    = 10.0_wp
    real(wp),parameter :: hun    = 100.0_wp

    public :: daxpy,dcopy,ddot,dscal

    interface
      function blas_c_fma(a,b,c)bind(C,name='fma')result(v)
       import c_double
       real(c_double),value::a,b,c
       real(c_double)::v
      end function
    end interface
    contains
!*******************************************************************************

!*******************************************************************************
!>
!  constant times a vector plus a vector.
!  uses unrolled loops for increments equal to one.
!
!### Author
!  jack dongarra, linpack, 3/11/78.

      subroutine daxpy(n,da,dx,incx,dy,incy)
      implicit none

      integer,intent(in) :: n
      real(wp),intent(in) :: da
      real(wp),intent(in)  :: dx(*)
      integer,intent(in)  :: incx
      real(wp),intent(inout) :: dy(*)
      integer,intent(in)  :: incy

      integer :: i , ix , iy , m , mp1

      if ( n<=0 ) return
      if ( da==zero ) return
      if ( incx==1 .and. incy==1 ) then

        ! code for both increments equal to 1

        ! clean-up loop

         m = mod(n,4)
         if ( m/=0 ) then
            do i = 1 , m
               dy(i) = blas_fma(da,dx(i),dy(i))
            end do
            if ( n<4 ) return
         end if
         mp1 = m + 1
         do i = mp1 , n , 4
            dy(i) = blas_fma(da,dx(i),dy(i))
            dy(i+1) = blas_fma(da,dx(i+1),dy(i+1))
            dy(i+2) = blas_fma(da,dx(i+2),dy(i+2))
            dy(i+3) = blas_fma(da,dx(i+3),dy(i+3))
         end do

      else

         ! code for unequal increments or equal increments
         ! not equal to 1

         ix = 1
         iy = 1
         if ( incx<0 ) ix = (-n+1)*incx + 1
         if ( incy<0 ) iy = (-n+1)*incy + 1
         do i = 1 , n
            dy(iy) = blas_fma(da,dx(ix),dy(iy))
            ix = ix + incx
            iy = iy + incy
         end do

      end if

      end subroutine daxpy
!*******************************************************************************

!*******************************************************************************
!>
!  copies a vector, x, to a vector, y.
!  uses unrolled loops for increments equal to one.
!
!### Author
!  jack dongarra, linpack, 3/11/78.

    subroutine dcopy(n,dx,incx,dy,incy)

      implicit none

      integer,intent(in) :: n
      real(wp),intent(in) :: dx(*)
      integer,intent(in) :: incx
      real(wp),intent(inout) :: dy(*)
      integer,intent(in) :: incy

      integer :: i , ix , iy , m , mp1

      if ( n<=0 ) return
      if ( incx==1 .and. incy==1 ) then

         ! code for both increments equal to 1

         ! clean-up loop

         m = mod(n,7)
         if ( m/=0 ) then
            do i = 1 , m
               dy(i) = dx(i)
            end do
            if ( n<7 ) return
         end if
         mp1 = m + 1
         do i = mp1 , n , 7
            dy(i) = dx(i)
            dy(i+1) = dx(i+1)
            dy(i+2) = dx(i+2)
            dy(i+3) = dx(i+3)
            dy(i+4) = dx(i+4)
            dy(i+5) = dx(i+5)
            dy(i+6) = dx(i+6)
         end do

      else

         ! code for unequal increments or equal increments
         ! not equal to 1

         ix = 1
         iy = 1
         if ( incx<0 ) ix = (-n+1)*incx + 1
         if ( incy<0 ) iy = (-n+1)*incy + 1
         do i = 1 , n
            dy(iy) = dx(ix)
            ix = ix + incx
            iy = iy + incy
         end do

      end if

    end subroutine dcopy
!*******************************************************************************

!*******************************************************************************
!>
!  forms the dot product of two vectors.
!  uses unrolled loops for increments equal to one.
!
!### Author
!  jack dongarra, linpack, 3/11/78.

      real(wp) function ddot(n,dx,incx,dy,incy)

      implicit none

      integer,intent(in) :: n
      real(wp),intent(in) :: dx(*)
      integer,intent(in) :: incx
      real(wp),intent(in) :: dy(*)
      integer,intent(in) :: incy

      real(wp) :: dtemp
      integer :: i , ix , iy , m , mp1

      ddot = zero
      dtemp = zero
      if ( n<=0 ) return
      if ( incx==1 .and. incy==1 ) then

         if(wp==c_double)then
           ddot=skx_dot(dx(1:n),dy(1:n));return
         end if
         ! code for both increments equal to 1

         ! clean-up loop

         m = mod(n,5)
         if ( m/=0 ) then
            do i = 1 , m
               dtemp = dtemp + dx(i)*dy(i)
            end do
            if ( n<5 ) then
               ddot = dtemp
               return
            end if
         end if
         mp1 = m + 1
         do i = mp1 , n , 5
            dtemp = dtemp + dx(i)*dy(i) + dx(i+1)*dy(i+1) + &
                    dx(i+2)*dy(i+2) + dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
         end do
         ddot = dtemp

      else

         ! code for unequal increments or equal increments
         ! not equal to 1

         ix = 1
         iy = 1
         if ( incx<0 ) ix = (-n+1)*incx + 1
         if ( incy<0 ) iy = (-n+1)*incy + 1
         do i = 1 , n
            dtemp = dtemp + dx(ix)*dy(iy)
            ix = ix + incx
            iy = iy + incy
         end do
         ddot = dtemp

      end if

      end function ddot
!*******************************************************************************

!*******************************************************************************
!>
!  scales a vector by a constant.
!  uses unrolled loops for increment equal to one.
!
!### Author
!  jack dongarra, linpack, 3/11/78.

    subroutine dscal(n,da,dx,incx)

      implicit none

      integer,intent(in) :: n
      real(wp),intent(in) :: da
      real(wp),intent(inout) :: dx(*)
      integer,intent(in) :: incx

      integer :: i , m , mp1 , nincx

      if ( n<=0 .or. incx<=0 ) return
      if ( incx==1 ) then

         ! code for increment equal to 1

         ! clean-up loop

         m = mod(n,5)
         if ( m/=0 ) then
            do i = 1 , m
               dx(i) = da*dx(i)
            end do
            if ( n<5 ) return
         end if
         mp1 = m + 1
         do i = mp1 , n , 5
            dx(i) = da*dx(i)
            dx(i+1) = da*dx(i+1)
            dx(i+2) = da*dx(i+2)
            dx(i+3) = da*dx(i+3)
            dx(i+4) = da*dx(i+4)
         end do
      else

         ! code for increment not equal to 1

         nincx = n*incx
         do i = 1 , nincx , incx
            dx(i) = da*dx(i)
         end do

      end if

      end subroutine dscal
!*******************************************************************************

#endif

!*******************************************************************************
 real(wp)function blas_fma(a,b,c)result(value)
  real(wp),intent(in)::a,b,c
  if(wp==c_double)then
   value=real(blas_c_fma(real(a,c_double),real(b,c_double),real(c,c_double)),wp)
  else
   value=a*b+c
  end if
 end function
 ! Real64 DDOT accumulation follows OpenBLAS SkylakeX. Copyright (c) 2014,
 ! The OpenBLAS Project; retained BSD notices in LICENSES/OpenBLAS-LICENSE.
 real(wp) function skx_dot(x,y) result(value)
  real(wp),intent(in)::x(:),y(:)
  real(wp)::wide(8,4),narrow(4,4),v(4)
  integer::n,n16,n32,i,j,k,index
  n=size(x);value=0
  if(n==1) then
   value=x(1)*y(1);return
  end if
  n16=n-mod(n,16);n32=n16-mod(n16,32);wide=0;narrow=0
  do i=1,n32,32
   do j=1,4
    do k=1,8
     index=i+(j-1)*8+k-1;wide(k,j)=blas_fma(x(index),y(index),wide(k,j))
    end do
   end do
  end do
  do j=1,4
   do k=1,4
    narrow(k,j)=wide(k,j)+wide(k+4,j)
   end do
  end do
  if(n16>n32) then
   do j=1,4
    do k=1,4
     index=n32+(j-1)*4+k;narrow(k,j)=blas_fma(x(index),y(index),narrow(k,j))
    end do
   end do
  end if
  if(n16>0) then
   do k=1,4
    v(k)=((narrow(k,1)+narrow(k,2))+narrow(k,3))+narrow(k,4)
   end do
   value=(v(1)+v(3))+(v(2)+v(4))
  end if
  do i=n16+1,n
   value=blas_fma(x(i),y(i),value)
  end do
 end function
    end module lbfgsb_blas_module
!*******************************************************************************
