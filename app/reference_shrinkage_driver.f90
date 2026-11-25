program reference_shrinkage_driver
  use sklearn_common,only:dp,require
  use sklearn_shrinkage,only:ledoit_wolf_shrinkage
  use sklearn_covariance,only:ledoit_wolf
  implicit none
  real(dp),allocatable::x(:,:),copy(:,:)
  integer::n,p,block,centered,i,ios,fit_model
  type(ledoit_wolf)::model
  real(dp)::shrinkage
  do
    read(*,*,iostat=ios)n,p,block,centered,fit_model
    if(ios/=0)exit
    allocate(x(n,p))
    do i=1,n
      read(*,*)x(i,:)
    end do
    copy=x
    shrinkage=ledoit_wolf_shrinkage(x,centered/=0,block)
    call require(all(x==copy),'Ledoit-Wolf changed input')
    write(*,'(es26.17e3)')shrinkage
    if(fit_model/=0)then
      model%block_size=block;model%assume_centered=centered/=0
      call model%fit(x)
      write(*,'(es26.17e3)')model%shrinkage_
      write(*,'(*(es26.17e3,1x))')model%location_
      do i=1,p
        write(*,'(*(es26.17e3,1x))')model%covariance_(i,:)
      end do
      do i=1,p
        write(*,'(*(es26.17e3,1x))')model%precision_(i,:)
      end do
    end if
    deallocate(x)
  end do
end program
