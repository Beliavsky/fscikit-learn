program test_bicluster_datasets
  use sklearn_common, only: dp,require
  use sklearn_datasets, only: make_biclusters,make_checkerboard
  implicit none
  real(dp), allocatable :: x(:,:),other(:,:)
  logical, allocatable :: rows(:,:),cols(:,:),r2(:,:),c2(:,:)
  integer, allocatable :: cover(:,:),before(:),after(:)
  integer :: i,j,k,nseed
  call make_biclusters([47,33],3,x,rows,cols,shuffle=.false.,random_state=42)
  call require(all(count(rows,dim=1)==1).and.all(count(cols,dim=1)==1),'diagonal cluster memberships partition axes')
  do i=1,size(x,1)
    do j=1,size(x,2)
      if(any(rows(:,i).and.cols(:,j))) then
        call require(x(i,j)>=10.and.x(i,j)<=100,'diagonal blocks lie within amplitude bounds')
      else
        call require(x(i,j)==0,'off-diagonal cells remain zero')
      end if
    end do
  end do
  call make_checkerboard([31,23],[2,3],x,rows,cols,minval=7._dp,maxval=7._dp,random_state=7)
  call require(all(shape(rows)==[6,31]).and.all(shape(cols)==[6,23]),'rectangular checkerboard returns Cartesian blocks')
  call require(all(x==7),'equal amplitude bounds give a constant checkerboard')
  allocate(cover(31,23));cover=0
  do k=1,6
    do i=1,31
      do j=1,23
        if(rows(k,i).and.cols(k,j)) cover(i,j)=cover(i,j)+1
      end do
    end do
  end do
  call require(all(cover==1),'Cartesian block masks cover each cell exactly once')
  call make_biclusters([1,1],8,x,rows,cols,random_state=0)
  call require(size(rows,1)==8.and.count(rows)==1.and.count(cols)==1,'more clusters than axis items preserve empty blocks')
  call make_checkerboard([0,3],[2,3],x,rows,cols,random_state=7)
  call require(all(shape(x)==[0,3]).and.all(shape(rows)==[6,0]).and.all(shape(cols)==[6,3]), &
    'empty matrix axes preserve membership shapes')
  call make_biclusters([0,0],2,x,rows,cols)
  call require(size(x)==0.and.size(rows)==0.and.size(cols)==0,'fully empty diagonal data')
  call make_biclusters([150,130],2,x,rows,cols,noise=.3_dp,random_state=42)
  call make_biclusters([150,130],2,other,r2,c2,noise=.3_dp,random_state=42)
  call require(all(x==other).and.all(rows.eqv.r2).and.all(cols.eqv.c2),'BTPE-scale seeded reproducibility')
  call random_seed(size=nseed);allocate(before(nseed),after(nseed));call random_seed(get=before)
  call make_checkerboard([17,21],3,x,rows,cols,random_state=42)
  call random_seed(get=after);call require(all(before==after),'bicluster generators leave intrinsic RNG unchanged')
  print *, 'Bicluster generators, Cartesian memberships, empty axes and multinomial streams: PASS'
end program
