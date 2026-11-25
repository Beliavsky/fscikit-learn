! Ledoit-Wolf shrinkage adapted from sklearn 1.6.1 covariance/_shrunk_covariance.py.
! Retain sklearn BSD notices in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_shrinkage
  use sklearn_common,only:dp,require,check_x
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::ledoit_wolf_shrinkage
contains
  real(dp)function ledoit_wolf_shrinkage(x,assume_centered,block_size)result(shrinkage)
    real(dp),intent(in)::x(:,:)
    logical,optional,intent(in)::assume_centered
    integer,optional,intent(in)::block_size
    real(dp),allocatable::centered(:,:),squares(:,:),trace(:)
    real(dp)::mu,beta_sum,delta_sum,beta,delta
    integer::n,p,block,splits,i,j
    logical::center,modes(size(ieee_all)),flags(size(ieee_all))
    call check_x(x)
    block=1000;if(present(block_size))block=block_size
    call require(block>=1,'Ledoit-Wolf block_size must be positive')
    n=size(x,1);p=size(x,2);shrinkage=0
    if(p==1)return
    center=.true.;if(present(assume_centered))center=.not.assume_centered
    ! Windows mode changes can clear status: save every flag before all modes.
    call ieee_get_flag(ieee_all,flags)
    call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
    centered=x
    if(center)centered=centered-spread(sum(centered,dim=1)/n,1,n)
    squares=centered**2;trace=sum(squares,dim=1)/n;mu=sum(trace)/p
    splits=p/block;beta_sum=0;delta_sum=0
    do i=0,splits-1
      do j=0,splits-1
        call accumulate(i*block+1,(i+1)*block,j*block+1,(j+1)*block)
      end do
      call accumulate(i*block+1,(i+1)*block,splits*block+1,p)
    end do
    do j=0,splits-1
      call accumulate(splits*block+1,p,j*block+1,(j+1)*block)
    end do
    call accumulate(splits*block+1,p,splits*block+1,p)
    delta_sum=delta_sum/(real(n,dp)*n)
    beta=(beta_sum/n-delta_sum)/(real(p,dp)*n)
    delta=(delta_sum-2*mu*sum(trace)+p*mu**2)/p
    if(delta<beta)beta=delta
    if(beta/=0)shrinkage=beta/delta
    call ieee_set_halting_mode(ieee_all,modes)
    call ieee_set_flag(ieee_all,flags)
  contains
    subroutine accumulate(first,last,other_first,other_last)
      integer,intent(in)::first,last,other_first,other_last
      real(dp),allocatable::product(:,:)
      if(last<first.or.other_last<other_first)return
      product=matmul(transpose(squares(:,first:last)),squares(:,other_first:other_last))
      beta_sum=beta_sum+sum(product)
      product=matmul(transpose(centered(:,first:last)),centered(:,other_first:other_last))
      delta_sum=delta_sum+sum(product**2)
    end subroutine
  end function
end module
