program reference_bicluster_metrics_driver
  use sklearn_metrics, only: consensus_score
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: count,k,na,nac,nb,nbc,ra,ca,rb,cb,mode,representation,i
  real(dp), allocatable :: ar(:,:),ac(:,:),br(:,:),bc(:,:),table(:,:)
  real(dp) :: value
  read(*,*) count
  do k=1,count
    read(*,*) na,nac,nb,nbc,ra,ca,rb,cb,mode,representation
    allocate(ar(na,ra),ac(nac,ca),br(nb,rb),bc(nbc,cb))
    read(*,*) ar;read(*,*) ac;read(*,*) br;read(*,*) bc
    if(mode==1) then
      allocate(table(na,nb));read(*,*) table
      do i=1,na
        ar(i,1)=real(i,dp)
      end do
      do i=1,nb
        br(i,1)=real(i,dp)
      end do
      value=consensus_score(ar,ac,br,bc,similarity=similarity)
      deallocate(table)
    else
      select case(representation)
      case(1)
        value=consensus_score(ar,ac,br,bc)
      case(2)
        value=consensus_score(nint(ar),nint(ac),nint(br),nint(bc))
      case(3)
        value=consensus_score(ar/=0,ac/=0,br/=0,bc/=0)
      case(4)
        value=consensus_score(ar,nint(ac),br/=0,int(bc,int64))
      end select
    end if
    write(*,'(ES26.17E3)') value
    deallocate(ar,ac,br,bc)
  end do
contains
  real(dp) function similarity(a_rows,a_columns,b_rows,b_columns) result(value)
    real(dp), intent(in) :: a_rows(:),a_columns(:),b_rows(:),b_columns(:)
    value=table(nint(a_rows(1)),nint(b_rows(1)))
  end function
end program
